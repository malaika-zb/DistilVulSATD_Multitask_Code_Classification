"""Phase 2: fine-tune a multitask teacher (default CodeBERT) on Colab T4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distilvulsatd.data.dataset import (  # noqa: E402
    carve_train_val,
    labels_to_array,
    load_competition_frames,
    save_json,
)
from distilvulsatd.data.preprocess import format_input  # noqa: E402
from distilvulsatd.evaluate import evaluate_model, print_eval_result  # noqa: E402
from distilvulsatd.models.losses import AsymmetricFocalLoss, inverse_freq_weights  # noqa: E402
from distilvulsatd.models.multitask import DualHeadMultitaskModel, load_sequence_classifier  # noqa: E402


class FunctionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def build_texts(df, mode: str):
    return [format_input(t, mode=mode) for t in df["Function"].tolist()]


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device, scaler):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
        total += float(loss.item())
    return total / max(len(loader), 1)


@torch.inference_mode()
def eval_loss(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total += float(loss_fn(outputs.logits, labels).item())
    return total / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "NLBSE_2027_Dataset"))
    parser.add_argument("--model-name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--head", choices=["shared", "dual"], default="shared")
    parser.add_argument("--input-mode", choices=["raw", "bimodal"], default="raw")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--loss", choices=["bce", "focal"], default="focal")
    parser.add_argument("--gamma-pos", type=float, default=0.0)
    parser.add_argument("--gamma-neg", type=float, default=3.0)
    parser.add_argument("--satd-boost", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Where to save checkpoints/metrics. On Colab, defaults to Google Drive.",
    )
    parser.add_argument(
        "--drive-root",
        type=str,
        default="/content/drive/MyDrive/NLBSE27_DistilVulSATD",
        help="Persistent Google Drive project root for Colab runs.",
    )
    parser.add_argument(
        "--eval-public-test",
        action="store_true",
        help="Also score once on public test (use sparingly).",
    )
    parser.add_argument("--quick", action="store_true", help="Tiny subset smoke train.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device=", device)

    frames = load_competition_frames(args.data_dir)
    train_df = frames["train"]
    test_df = frames["test"]
    if args.quick:
        train_df = train_df.sample(n=2000, random_state=args.seed).reset_index(drop=True)
        args.epochs = 1
        args.eval_public_test = False

    fit_df, val_df = carve_train_val(train_df, val_fraction=args.val_fraction, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    fit_texts = build_texts(fit_df, args.input_mode)
    val_texts = build_texts(val_df, args.input_mode)
    fit_labels = labels_to_array(fit_df)
    val_labels = labels_to_array(val_df)

    fit_ds = FunctionDataset(fit_texts, fit_labels, tokenizer, args.max_length)
    val_ds = FunctionDataset(val_texts, val_labels, tokenizer, args.max_length)
    fit_loader = DataLoader(fit_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    if args.head == "shared":
        model = load_sequence_classifier(args.model_name)
    else:
        model = DualHeadMultitaskModel(args.model_name)
    model.to(device)

    vuln_rate = float(fit_df["vuln"].mean())
    satd_rate = float(fit_df["satd"].mean())
    weights = inverse_freq_weights(vuln_rate, satd_rate, satd_boost=args.satd_boost).to(device)

    if args.loss == "focal":
        loss_fn = AsymmetricFocalLoss(
            gamma_pos=args.gamma_pos,
            gamma_neg=args.gamma_neg,
            class_weights=weights,
        )
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(len(fit_loader) * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    run_name = (
        f"{args.model_name.split('/')[-1]}_{args.head}_{args.input_mode}_"
        f"L{args.max_length}_{args.loss}_e{args.epochs}"
    )
    if args.out_dir:
        base_out = Path(args.out_dir)
    elif Path("/content/drive/MyDrive").exists() or Path(args.drive_root).exists():
        # Prefer persistent Drive over ephemeral /content
        try:
            from distilvulsatd.colab_paths import setup_colab_drive

            drive_root = setup_colab_drive(args.drive_root)
        except Exception:
            drive_root = Path(args.drive_root)
            (drive_root / "checkpoints").mkdir(parents=True, exist_ok=True)
            (drive_root / "results").mkdir(parents=True, exist_ok=True)
        base_out = drive_root / "checkpoints" / "teacher_runs"
    else:
        base_out = ROOT / "results" / "teacher_runs"
    out_dir = base_out / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Saving run to:", out_dir)

    history = []
    best_val_f1 = -1.0
    best_path = out_dir / "best_model"

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, fit_loader, optimizer, scheduler, loss_fn, device, scaler)
        va_loss = eval_loss(model, val_loader, loss_fn, device)

        # Fast val metrics without FLOPs profiling
        val_result = evaluate_model(
            model,
            tokenizer,
            val_texts,
            val_labels,
            device=device,
            batch_size=min(64, args.batch_size * 2),
            max_length=args.max_length,
            repetitions=1,
            measure_flops=False,
            show_progress=False,
        )
        print(
            f"epoch {epoch}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"val_avgF1={val_result.avg_f1:.4f}"
        )
        print_eval_result(val_result, title=f"VAL epoch {epoch}")
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                **val_result.to_dict(),
            }
        )
        if val_result.avg_f1 > best_val_f1:
            best_val_f1 = val_result.avg_f1
            model_to_save = model
            if hasattr(model, "save_pretrained"):
                model_to_save.save_pretrained(best_path)
            else:
                torch.save(model.state_dict(), best_path.with_suffix(".pt"))
            tokenizer.save_pretrained(best_path)
            print(f"saved best -> {best_path}")

    payload = {
        "args": vars(args),
        "best_val_f1": best_val_f1,
        "history": history,
        "vuln_rate": vuln_rate,
        "satd_rate": satd_rate,
        "class_weights": weights.detach().cpu().tolist(),
    }

    if args.eval_public_test:
        # reload best shared head if possible
        if args.head == "shared" and best_path.exists():
            model = load_sequence_classifier(str(best_path))
            model.to(device)
        test_texts = build_texts(test_df, args.input_mode)
        test_result = evaluate_model(
            model,
            tokenizer,
            test_texts,
            labels_to_array(test_df),
            device=device,
            batch_size=600 if device.type == "cuda" else 16,
            max_length=args.max_length,
            repetitions=10 if device.type == "cuda" else 1,
            measure_flops=device.type == "cuda",
        )
        print_eval_result(test_result, title="PUBLIC TEST")
        payload["public_test"] = test_result.to_dict()

    save_json(payload, out_dir / "metrics.json")
    print(f"Wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
