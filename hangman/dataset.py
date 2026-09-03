"""GPU-resident state store.

The DAgger buffer holds a few million game states. They are small and fixed
width, so instead of a ``DataLoader`` with host-to-device copies we keep the
whole buffer on the GPU in compact dtypes and slice it by index. That removes
the input pipeline as a bottleneck entirely: batches cost a gather.

Memory for 3M states: board 3M x 29 uint8 (87 MB) + three int32 columns
(36 MB) = well within an 8 GB card alongside the model.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import N_LETTERS
from .simulator import StateBuffer


def unpack_bitmask(masks: torch.Tensor) -> torch.Tensor:
    """``(B,)`` int32 bitmask -> ``(B, 26)`` bool."""
    bits = torch.arange(N_LETTERS, device=masks.device, dtype=torch.int32)
    return ((masks.unsqueeze(1) >> bits) & 1).bool()


class GpuStateStore:
    """Holds a :class:`StateBuffer` on device and yields training batches."""

    def __init__(
        self,
        buffer: StateBuffer,
        encoded_corpus: np.ndarray,
        device: torch.device | str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.board = torch.from_numpy(np.ascontiguousarray(buffer.board)).to(self.device)
        self.guessed = torch.from_numpy(
            np.ascontiguousarray(buffer.guessed, dtype=np.int32)
        ).to(self.device)
        self.contains = torch.from_numpy(
            np.ascontiguousarray(buffer.contains, dtype=np.int32)
        ).to(self.device)
        self.word_id = torch.from_numpy(
            np.ascontiguousarray(buffer.word_id, dtype=np.int64)
        ).to(self.device)
        #: Full corpus, used to recover the true letter behind every hidden slot.
        self.corpus = torch.from_numpy(np.ascontiguousarray(encoded_corpus)).to(self.device)

    def __len__(self) -> int:
        return int(self.board.shape[0])

    def batch(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        board = self.board.index_select(0, idx).long()
        guessed = unpack_bitmask(self.guessed.index_select(0, idx))
        contains = unpack_bitmask(self.contains.index_select(0, idx))
        truth = self.corpus.index_select(0, self.word_id.index_select(0, idx)).long()

        # Target: letters that are in the word and have not been guessed yet.
        target = (contains & ~guessed).float()
        return {"board": board, "guessed": guessed, "target": target, "truth": truth}

    def epoch_batches(self, batch_size: int, generator: torch.Generator | None = None):
        """Yield shuffled batches covering the whole store once."""
        order = torch.randperm(len(self), device=self.device, generator=generator)
        for start in range(0, len(self), batch_size):
            yield self.batch(order[start : start + batch_size])
