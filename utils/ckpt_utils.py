import torch
import os
import pathlib
from typing import Dict, Any, Optional

def save_checkpoint(run_dir, epoch, model, optimizer, scheduler, val_loss, best=False, ema_model=None):
    state = {
        "epoch": epoch,
        "ema_model_state": ema_model.state_dict() if ema_model else None,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "val_loss": val_loss,
    }
    ckpt_name = "best.pt" if best else "last.pt"
    torch.save(state, os.path.join(run_dir, "checkpoints", ckpt_name))


def load_best_checkpoint(run_dir, model, ema_model=None):
    ckpt_name = "best.pt"
    path = os.path.join(run_dir, "checkpoints", ckpt_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")

    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    if ema_model is not None:
        ema_model.load_state_dict(state["ema_model_state"])