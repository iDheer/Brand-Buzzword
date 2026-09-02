"""Masked-word Transformer for Hangman letter prediction.

Modelling view
--------------
A Hangman decision is a *set prediction* problem conditioned on a partially
observed string:

    given   board  = "_ a _ i n g"  and  ruled-out = {e, o, t}
    predict P(letter c occurs among the hidden slots)  for every c

The network answers that question with two complementary heads:

``set head``
    Pools the encoded board into a single vector and emits 26 presence logits
    directly. Good at global, morphology-level cues ("this looks like a
    ``-ation`` word, so ``t`` is likely").

``position head``
    Emits a distribution over letters *for every hidden slot*, then combines
    them with a noisy-OR into a presence probability. Good at local, spelling
    level cues ("slot 3 sits between ``a`` and ``i``, so it is probably ``r``").

A learned fusion layer blends the two. Both heads exploit a hard constraint
that follows from the rules: because a correct guess reveals *all* of its
occurrences at once, a slot that is still hidden can never hold a letter that
has already been guessed. That mask is applied inside the position head, so
the model never has to learn it from data.
"""

from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MASK_TOKEN, MAX_WORD_LEN, N_LETTERS, PAD_TOKEN, VOCAB_SIZE, ModelConfig


class GameStateFeatures(nn.Module):
    """Derives every auxiliary feature from ``(board, guessed)`` alone.

    Keeping this inside the model guarantees that training and inference see
    byte-identical inputs -- there is no separate feature pipeline to drift.
    """

    n_context_features = 2 * N_LETTERS + 6

    @staticmethod
    def forward(board: torch.Tensor, guessed: torch.Tensor) -> dict[str, torch.Tensor]:
        is_pad = board == PAD_TOKEN
        is_hidden = board == MASK_TOKEN
        is_revealed = ~(is_pad | is_hidden)

        lengths = (~is_pad).sum(dim=1, keepdim=True).float()
        n_hidden = is_hidden.sum(dim=1, keepdim=True).float()

        # Which letters are currently visible on the board.
        letters_on_board = torch.zeros_like(guessed, dtype=torch.bool)
        safe = torch.where(is_revealed, board, torch.full_like(board, N_LETTERS))
        scatter_target = torch.zeros(
            (board.shape[0], N_LETTERS + 1), dtype=torch.bool, device=board.device
        )
        scatter_target.scatter_(1, safe, True)
        letters_on_board = scatter_target[:, :N_LETTERS]

        # A guessed letter that is not on the board was a miss: it is absent.
        absent = guessed & ~letters_on_board

        n_absent = absent.sum(dim=1, keepdim=True).float()
        n_guessed = guessed.sum(dim=1, keepdim=True).float()

        scalars = torch.cat(
            [
                lengths / MAX_WORD_LEN,
                n_hidden / MAX_WORD_LEN,
                n_hidden / lengths.clamp_min(1.0),
                n_absent / 6.0,
                n_guessed / N_LETTERS,
                letters_on_board.sum(dim=1, keepdim=True).float() / N_LETTERS,
            ],
            dim=1,
        )

        context = torch.cat([absent.float(), letters_on_board.float(), scalars], dim=1)
        return {
            "context": context,
            "is_pad": is_pad,
            "is_hidden": is_hidden,
            "letters_on_board": letters_on_board,
            "absent": absent,
        }


class HangmanTransformer(nn.Module):
    """Transformer encoder over the masked word."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.token_embedding = nn.Embedding(VOCAB_SIZE, d, padding_idx=PAD_TOKEN)
        # Absolute position AND distance-from-the-end: suffix morphology
        # (-ing, -ness, -ation) is one of the strongest signals in this corpus,
        # and it is only expressible relative to the end of the word.
        self.forward_position = nn.Embedding(MAX_WORD_LEN + 1, d)
        self.reverse_position = nn.Embedding(MAX_WORD_LEN + 1, d)

        self.context_projection = nn.Sequential(
            nn.Linear(GameStateFeatures.n_context_features, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.input_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(cfg.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.encoder_norm = nn.LayerNorm(d)

        self.set_head = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, N_LETTERS),
        )
        self.position_head = nn.Linear(d, N_LETTERS)

        # Fusion of the two presence estimates, per letter.
        self.fusion_weight = nn.Parameter(torch.tensor([1.0, 1.0]))
        self.fusion_bias = nn.Parameter(torch.zeros(N_LETTERS))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    # ------------------------------------------------------------------ #

    def encode(self, board: torch.Tensor, guessed: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = GameStateFeatures.forward(board, guessed)
        batch, seq = board.shape

        positions = torch.arange(seq, device=board.device).unsqueeze(0).expand(batch, seq)
        # Distance from the end must be measured against the *extent* of the
        # word, not the count of non-pad tokens. Those differ exactly when the
        # word contains an interior non-letter (a space or digit in a brand
        # name, encoded as PAD): counting tokens would shift every preceding
        # slot's suffix distance by the number of such characters, corrupting
        # the strongest morphological signal the encoder has. The corpora are
        # pure a-z, where the two agree, so this is behaviour-preserving there.
        extent = (positions * (~feats["is_pad"])).max(dim=1, keepdim=True).values + 1
        reverse = (extent - 1 - positions).clamp(min=0, max=MAX_WORD_LEN)

        x = (
            self.token_embedding(board)
            + self.forward_position(positions.clamp(max=MAX_WORD_LEN))
            + self.reverse_position(reverse)
        )
        context = self.context_projection(feats["context"])
        x = self.input_norm(x + context.unsqueeze(1))
        x = self.dropout(x)

        hidden = self.encoder(x, src_key_padding_mask=feats["is_pad"])
        hidden = self.encoder_norm(hidden)
        feats["hidden"] = hidden
        feats["context_vector"] = context
        return feats

    def forward(
        self, board: torch.Tensor, guessed: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        feats = self.encode(board, guessed)
        hidden = feats["hidden"]
        valid = (~feats["is_pad"]).unsqueeze(-1).float()

        pooled_mean = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        pooled_max = hidden.masked_fill(feats["is_pad"].unsqueeze(-1), -1e4).max(dim=1).values
        pooled = torch.cat([pooled_mean, pooled_max, feats["context_vector"]], dim=1)

        set_logits = self.set_head(pooled)

        # --- position head -> noisy-OR presence probability ------------------
        slot_logits = self.position_head(hidden)
        # A hidden slot cannot hold an already-guessed letter (a correct guess
        # reveals every occurrence), so mask those out before the softmax.
        slot_logits = slot_logits.masked_fill(guessed.unsqueeze(1), -1e4)
        slot_probs = slot_logits.softmax(dim=-1)

        hidden_slots = feats["is_hidden"].unsqueeze(-1).float()
        miss_prob = 1.0 - slot_probs * hidden_slots  # 1.0 at non-hidden slots
        absent_prob = miss_prob.clamp(1e-6, 1.0).log().sum(dim=1).exp()
        present_prob = (1.0 - absent_prob).clamp(1e-6, 1.0 - 1e-6)
        noisy_or_logits = torch.log(present_prob) - torch.log1p(-present_prob)

        weight = self.fusion_weight
        fused_logits = (
            weight[0] * set_logits + weight[1] * noisy_or_logits + self.fusion_bias
        )

        return {
            "logits": fused_logits,
            "set_logits": set_logits,
            "noisy_or_logits": noisy_or_logits,
            "slot_logits": slot_logits,
            "is_hidden": feats["is_hidden"],
        }

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def as_policy(self, amp_dtype: torch.dtype | None = torch.bfloat16):
        """Return a ``Policy`` callable for :func:`hangman.simulator.play_games`."""
        self.eval()

        def policy(board: torch.Tensor, guessed: torch.Tensor) -> torch.Tensor:
            if amp_dtype is not None and board.is_cuda:
                with torch.autocast("cuda", dtype=amp_dtype):
                    return self(board, guessed)["logits"].float()
            return self(board, guessed)["logits"].float()

        return policy

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def hangman_loss(
    outputs: dict[str, torch.Tensor],
    target_present: torch.Tensor,
    guessed: torch.Tensor,
    truth: torch.Tensor,
    *,
    set_weight: float = 0.4,
    position_weight: float = 0.4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-task objective.

    ``fused``     -- the loss that actually matters: presence BCE on the fused
                     logits, evaluated only over letters still legal to guess.
    ``set``       -- same BCE on the raw set head, keeping it independently useful.
    ``position``  -- cross-entropy on the true letter of every hidden slot,
                     which forces the encoder to learn spelling structure rather
                     than only bag-of-letters statistics.
    """
    legal = ~guessed  # already-guessed letters carry no decision value

    def presence_bce(logits: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits, target_present, reduction="none"
        )
        return (loss * legal).sum() / legal.sum().clamp_min(1.0)

    fused_loss = presence_bce(outputs["logits"])
    set_loss = presence_bce(outputs["set_logits"])

    slot_logits = outputs["slot_logits"]
    is_hidden = outputs["is_hidden"]
    if is_hidden.any():
        flat_logits = slot_logits[is_hidden]
        flat_target = truth[is_hidden].long()
        position_loss = F.cross_entropy(flat_logits, flat_target)
    else:
        position_loss = slot_logits.sum() * 0.0

    total = fused_loss + set_weight * set_loss + position_weight * position_loss
    stats = {
        "loss": float(total.detach()),
        "fused": float(fused_loss.detach()),
        "set": float(set_loss.detach()),
        "position": float(position_loss.detach()),
    }
    return total, stats
