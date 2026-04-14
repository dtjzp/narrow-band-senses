"""Compute bits-per-character on a test dataset with SEP masking."""

import math

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_bpc(
    model: torch.nn.Module,
    dataset,
    device: str = "cpu",
    batch_size: int = 64,
) -> float:
    """Compute bits-per-character on dataset, masking SEP tokens.

    Returns cross-entropy loss in bits (divided by ln(2)).
    Only counts positions where loss_mask == 1.
    """
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_chars = 0

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits = model(x)
        # Per-token cross-entropy (no reduction)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="none"
        )
        loss = loss.view_as(mask)
        # Apply mask and sum
        total_loss += (loss * mask).sum().item()
        total_chars += mask.sum().item()

    if total_chars == 0:
        return float("inf")
    # Convert nats to bits
    return (total_loss / total_chars) / math.log(2)
