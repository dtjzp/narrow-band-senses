"""Train S-size domain checkpoints for python_code, network, quantum.

Emits checkpoints in the format expected by phase_reverse_bridge_v2.load_s_model:
  {domain}_S_s42.pt with keys: state_dict, vocab, sep_token, n_layers, d_model,
  n_heads, context_len.

~2-3 minutes per domain on A100.
"""
from __future__ import annotations
import sys, os, time, json, math, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

SURVEY_CODE = '/content/drive/MyDrive/nbs-survey/code'
if SURVEY_CODE not in sys.path:
    sys.path.insert(0, SURVEY_CODE)

from config import MODEL_CONFIGS, TRAIN_CONFIG, DISCRETE_DOMAINS, TRAIN_FRAC, VAL_FRAC
from model import CharTransformer
from dataset import CharDataset, build_vocab

DATA_DIR = Path('/content/drive/MyDrive/nbs-survey/data')
CKPT_DIR = Path('/content/drive/MyDrive/nbs-bridge/checkpoints')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
CTX = 512
MODEL_SIZE = 'S'


def train_one(domain: str, max_epochs: int = 10, verbose: bool = True):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    out_path = CKPT_DIR / f'{domain}_S_s42.pt'
    if out_path.exists():
        print(f'[train_s] {domain}: checkpoint exists, skipping')
        return str(out_path)

    text_path = DATA_DIR / f'{domain}_1M.txt'
    if not text_path.exists():
        raise FileNotFoundError(f'missing training text: {text_path}')
    text = text_path.read_text(encoding='utf-8')
    is_discrete = domain in DISCRETE_DOMAINS
    vocab, inv_vocab = build_vocab(text, add_sep=is_discrete)
    sep_token = '<SEP>' if is_discrete else None

    n = len(text)
    tr_end = int(n * TRAIN_FRAC)
    va_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    train_ds = CharDataset(text[:tr_end], vocab, CTX, sep_token)
    val_ds   = CharDataset(text[tr_end:va_end], vocab, CTX, sep_token)

    mcfg = MODEL_CONFIGS[MODEL_SIZE]
    model = CharTransformer(
        vocab_size=len(vocab),
        n_layers=mcfg['n_layers'], d_model=mcfg['d_model'], n_heads=mcfg['n_heads'],
        context_len=CTX, dropout=TRAIN_CONFIG['dropout'],
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG['lr'],
                            weight_decay=TRAIN_CONFIG.get('weight_decay', 0.01))
    bs = TRAIN_CONFIG.get('batch_size', 64)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    best_val = float('inf')
    best_state = None
    history = []
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_loss, tr_n = 0.0, 0
        for batch in train_loader:
            x = batch[0].to(DEVICE) if isinstance(batch, (list, tuple)) else batch['input'].to(DEVICE)
            y = batch[1].to(DEVICE) if isinstance(batch, (list, tuple)) else batch['target'].to(DEVICE)
            m = batch[2].to(DEVICE) if isinstance(batch, (list, tuple)) and len(batch) > 2 else None
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction='none',
            )
            if m is not None:
                loss = (loss * m.reshape(-1)).sum() / m.sum().clamp(min=1)
            else:
                loss = loss.mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * x.size(0); tr_n += x.size(0)
        train_loss = tr_loss / max(tr_n, 1)

        model.eval()
        val_l, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(DEVICE) if isinstance(batch, (list, tuple)) else batch['input'].to(DEVICE)
                y = batch[1].to(DEVICE) if isinstance(batch, (list, tuple)) else batch['target'].to(DEVICE)
                m = batch[2].to(DEVICE) if isinstance(batch, (list, tuple)) and len(batch) > 2 else None
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    reduction='none',
                )
                if m is not None:
                    loss = (loss * m.reshape(-1)).sum() / m.sum().clamp(min=1)
                else:
                    loss = loss.mean()
                val_l += loss.item() * x.size(0); val_n += x.size(0)
        val_loss = val_l / max(val_n, 1)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        if verbose:
            print(f'[train_s/{domain}] e{epoch}: train={train_loss:.4f} val={val_loss:.4f}', flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save checkpoint in expected format
    ckpt = {
        'state_dict': best_state,
        'vocab': vocab,
        'sep_token': sep_token,
        'n_layers': mcfg['n_layers'],
        'd_model': mcfg['d_model'],
        'n_heads': mcfg['n_heads'],
        'context_len': CTX,
        'best_val_loss': best_val,
        'history': history,
        'elapsed_sec': time.time() - t0,
    }
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    bpc = best_val / math.log(2)
    print(f'[train_s/{domain}] DONE. best_val={best_val:.4f} bpc={bpc:.4f} saved={out_path}', flush=True)
    return str(out_path)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--domains', nargs='+', default=['python_code', 'network', 'quantum'])
    ap.add_argument('--epochs', type=int, default=10)
    args = ap.parse_args()
    for d in args.domains:
        try:
            train_one(d, max_epochs=args.epochs)
        except Exception as e:
            print(f'[train_s/{d}] FAILED: {type(e).__name__}: {e}', flush=True)
            import traceback; traceback.print_exc()
