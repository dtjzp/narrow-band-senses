"""Training loop with early stopping. Used by both Colab notebooks and local testing."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiment.code.config import (
    MODEL_CONFIGS, TRAIN_CONFIG, DISCRETE_DOMAINS, TARGET_CHARS,
    TRAIN_FRAC, VAL_FRAC,
)
from experiment.code.model import CharTransformer
from experiment.code.dataset import CharDataset, build_vocab, encode_text
from experiment.code.evaluate import compute_bpc


def train_one_run(
    domain: str,
    model_size: str,
    seed: int,
    context_len: int,
    data_dir: str,
    output_dir: str,
    max_epochs: int | None = None,
    device: str | None = None,
    checkpoint_dir: str | None = None,
    batch_size: int | None = None,
) -> dict:
    """Train one model on one domain and return results.

    Args:
        domain: Domain name (e.g., "english", "protein").
        model_size: One of "XS", "S", "M", "L".
        seed: Random seed for reproducibility.
        context_len: Context window length.
        data_dir: Directory containing {domain}_1M.txt files.
        output_dir: Directory to save results JSON.
        max_epochs: Override max epochs (for testing).
        device: "cuda", "cpu", or None (auto-detect).
        checkpoint_dir: If set, save/resume checkpoints here.
        batch_size: Override batch size (for OOM contingency on large configs).

    Returns:
        Dict with test_bpc, val_bpc, train_bpc, stopping_epoch, actual_params, etc.
    """
    # Seed everything
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = TRAIN_CONFIG.copy()
    if max_epochs is not None:
        cfg["max_epochs"] = max_epochs
    if batch_size is not None:
        cfg["batch_size"] = batch_size

    # Load data
    data_path = Path(data_dir) / f"{domain}_1M.txt"
    text = data_path.read_text(encoding="utf-8")

    # Build vocab
    is_discrete = domain in DISCRETE_DOMAINS
    vocab, inv_vocab = build_vocab(text, add_sep=is_discrete)
    sep_token = "<SEP>" if is_discrete else None

    # Split: train / val / test
    n = len(text)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    train_text = text[:train_end]
    val_text = text[train_end:val_end]
    test_text = text[val_end:]

    train_ds = CharDataset(train_text, vocab, context_len, sep_token)
    val_ds = CharDataset(val_text, vocab, context_len, sep_token)
    test_ds = CharDataset(test_text, vocab, context_len, sep_token)

    # Create model
    mcfg = MODEL_CONFIGS[model_size]
    model = CharTransformer(
        vocab_size=len(vocab),
        n_layers=mcfg["n_layers"],
        d_model=mcfg["d_model"],
        n_heads=mcfg["n_heads"],
        context_len=context_len,
        dropout=cfg["dropout"],
    ).to(device)

    actual_params = model.count_parameters()

    # Optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=cfg["betas"],
    )

    # Cosine schedule
    total_steps = cfg["max_epochs"] * max(len(train_ds) // cfg["batch_size"], 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=cfg["lr"] * 0.1
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

    # Training loop with early stopping
    best_val_bpc = float("inf")
    patience_counter = 0
    stopping_epoch = cfg["max_epochs"]
    best_state = None
    train_bpc = float("inf")

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_chars = 0

        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x)
            loss_per_token = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), reduction="none"
            ).view_as(mask)
            loss = (loss_per_token * mask).sum() / mask.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            epoch_loss += (loss_per_token * mask).sum().item()
            epoch_chars += mask.sum().item()

        train_bpc = (epoch_loss / max(epoch_chars, 1)) / math.log(2)
        val_bpc = compute_bpc(model, val_ds, device)

        # Checkpoint
        if checkpoint_dir:
            ckpt_path = Path(checkpoint_dir) / f"{domain}_{model_size}_s{seed}_ctx{context_len}_e{epoch}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

        # Early stopping
        if val_bpc < best_val_bpc - 1e-4:
            best_val_bpc = val_bpc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                stopping_epoch = epoch
                break

    # Restore best model and evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    test_bpc = compute_bpc(model, test_ds, device)

    result = {
        "domain": domain,
        "model_size": model_size,
        "seed": seed,
        "context_len": context_len,
        "actual_params": actual_params,
        "vocab_size": len(vocab),
        "test_bpc": test_bpc,
        "val_bpc": best_val_bpc,
        "train_bpc": train_bpc,
        "stopping_epoch": stopping_epoch,
        "max_epochs": cfg["max_epochs"],
    }

    # Save result
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fname = f"{domain}_{model_size}_s{seed}_ctx{context_len}.json"
    with open(out_path / fname, "w") as f:
        json.dump(result, f, indent=2)

    return result
