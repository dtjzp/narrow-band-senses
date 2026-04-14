"""Character-level dataset with SEP token masking.

Handles both continuous streams (English, code) and discrete sequences
(protein, SMILES) concatenated with <SEP> tokens.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


def build_vocab(text: str, add_sep: bool = False) -> tuple[dict[str, int], dict[int, str]]:
    """Build character vocabulary from text.

    For texts with <SEP> tokens, first strips them out before building
    the character vocab, then adds <SEP> as a special token.

    Returns (char_to_id, id_to_char) mappings.
    """
    # Strip <SEP> markers before building char vocab
    clean = text.replace("<SEP>", "")
    chars = sorted(set(c for c in clean if c != "\n"))
    vocab = {c: i for i, c in enumerate(chars)}
    if add_sep:
        vocab["<SEP>"] = len(vocab)
    inv_vocab = {v: k for k, v in vocab.items()}
    return vocab, inv_vocab


def encode_text(text: str, vocab: dict[str, int], sep_token: str | None = None) -> list[int]:
    """Encode text to integer sequence using vocabulary.

    If sep_token is set, replaces occurrences of sep_token with its vocab id.
    """
    if sep_token and sep_token in text:
        parts = text.split(sep_token)
        encoded = []
        for i, part in enumerate(parts):
            for c in part:
                if c in vocab:
                    encoded.append(vocab[c])
            if i < len(parts) - 1:
                encoded.append(vocab[sep_token])
        return encoded
    return [vocab[c] for c in text if c in vocab]


class CharDataset(Dataset):
    """Character-level dataset that returns (input, target, loss_mask) chunks.

    loss_mask is 0 at positions where the target is a SEP token,
    1 everywhere else. This ensures BPC is computed only on real characters.
    """

    def __init__(
        self,
        text: str,
        vocab: dict[str, int],
        context_len: int,
        sep_token: str | None = None,
    ):
        self.context_len = context_len
        self.sep_id = vocab.get(sep_token) if sep_token else None
        self.data = torch.tensor(encode_text(text, vocab, sep_token), dtype=torch.long)
        # Truncate to multiple of context_len
        n_chunks = len(self.data) // context_len
        self.data = self.data[: n_chunks * context_len]
        self.n_chunks = n_chunks

    def __len__(self):
        return max(0, self.n_chunks - 1)  # Need one extra token for targets

    def __getitem__(self, idx):
        start = idx * self.context_len
        end = start + self.context_len
        x = self.data[start:end]
        y = self.data[start + 1 : end + 1]

        # Loss mask: 0 where target is SEP, 1 elsewhere
        if self.sep_id is not None:
            mask = (y != self.sep_id).float()
        else:
            mask = torch.ones_like(y, dtype=torch.float)

        return x, y, mask
