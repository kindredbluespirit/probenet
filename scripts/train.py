"""Train baseline BC and ProbeNet-BC policies on sim data.

Usage:
    python scripts/train.py --data-dir data/sim --output-dir outputs/models [--variant baseline|probenet]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from probenet.conditioning import MLPConditioner
from probenet.dataset import create_loaders
from probenet.policies import BCPolicy, ProbeNetPolicy


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    count = 0
    for batch in loader:
        rgb = batch["rgb"].to(device)
        state = batch["state"].to(device)
        action = batch["action"].to(device)
        visual_params = batch["visual_params"].to(device)
        physical_params = batch["physical_params"].to(device)

        if isinstance(model, ProbeNetPolicy):
            pred = model(rgb, state, visual_params, physical_params)
        else:
            pred = model(rgb, state)

        loss = nn.functional.mse_loss(pred, action)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in loader:
        rgb = batch["rgb"].to(device)
        state = batch["state"].to(device)
        action = batch["action"].to(device)
        visual_params = batch["visual_params"].to(device)
        physical_params = batch["physical_params"].to(device)

        if isinstance(model, ProbeNetPolicy):
            pred = model(rgb, state, visual_params, physical_params)
        else:
            pred = model(rgb, state)

        total_loss += nn.functional.mse_loss(pred, action).item()
        count += 1
    return total_loss / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ProbeNet policies.")
    parser.add_argument("--data-dir", type=str, default="data/sim")
    parser.add_argument("--output-dir", type=str, default="outputs/models")
    parser.add_argument("--variant", type=str, default="both", choices=["baseline", "probenet", "both"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = create_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        train_split=0.8,
        device=device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = ["baseline", "probenet"] if args.variant == "both" else [args.variant]

    for variant in variants:
        print(f"\n{'='*60}\nTraining {variant}\n{'='*60}")

        if variant == "baseline":
            model = BCPolicy(action_dim=6).to(device)
        else:
            conditioner = MLPConditioner(visual_dim=4, physical_dim=3, output_dim=32)
            model = ProbeNetPolicy(action_dim=6, conditioner=conditioner).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val_loss = float("inf")
        history: list[dict] = []

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_loss = evaluate(model, val_loader, device)

            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = output_dir / f"{variant}_best.pt"
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt)

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | train loss: {train_loss:.6f} | val loss: {val_loss:.6f}")

        with open(output_dir / f"{variant}_history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"Best val loss: {best_val_loss:.6f} — saved to {output_dir / f'{variant}_best.pt'}")


if __name__ == "__main__":
    main()
