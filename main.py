#!/usr/bin/env python3
"""
Main entry point for deepfake detection pipeline.

Usage:
  python main.py train   --data_dir /path/to/data --epochs 30
  python main.py eval    --data_dir /path/to/data --model checkpoints/best_model.pth
  python main.py predict --input  /path/to/image.jpg
  python main.py prepare --data_dir /path/to/data
  python main.py serve   --port 8000
"""

import argparse
import json
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_train(args):
    from src.train import Trainer, DeepfakeDataset, get_default_config

    config = get_default_config()
    config.update({
        "epochs":     args.epochs,
        "batch_size": args.batch_size,
        "lr":         args.lr,
        "save_dir":   args.save_dir,
    })

    logger.info("Config: " + json.dumps(config, indent=2))

    train_set = DeepfakeDataset(args.data_dir, split="train", augment=True)
    val_set   = DeepfakeDataset(args.data_dir, split="val",   augment=False)

    trainer = Trainer(config)
    history = trainer.train(train_set, val_set)

    logger.info(f"Best Val AUC: {max(history['val_auc']):.4f}")


def cmd_eval(args):
    from src.inference import DeepfakeInference
    from src.train import DeepfakeDataset
    from utils.evaluate import evaluate_on_dataset, plot_roc_curve, plot_pr_curve

    engine = DeepfakeInference(
        model_path=args.model,
        generate_heatmap=False,
    )
    dataset = DeepfakeDataset(args.data_dir, split="test", augment=False)

    logger.info(f"Evaluating on {len(dataset)} test samples...")
    metrics = evaluate_on_dataset(engine, dataset)

    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        if k != "classification_report":
            print(f"  {k:25s}: {v}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {out_dir}/metrics.json")


def cmd_predict(args):
    from src.inference import DeepfakeInference
    import numpy as np

    engine = DeepfakeInference(
        model_path=args.model,
        generate_heatmap=not args.no_heatmap,
    )

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    suffix = input_path.suffix.lower()
    if suffix in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        result = engine.predict_video(str(input_path))
        print(f"\n  VIDEO: {args.input}")
        print(f"  Frames analyzed : {len(result.frame_results)}")
    else:
        result = engine.predict_image(str(input_path))
        print(f"\n  IMAGE: {args.input}")
        if result.heatmap is not None and args.save_heatmap:
            from PIL import Image
            Image.fromarray(result.heatmap).save(args.save_heatmap)
            print(f"  Heatmap saved   : {args.save_heatmap}")

    print(f"\n  ┌─ RESULT ──────────────────────────────")
    print(f"  │  Label       : {result.label}")
    print(f"  │  Probability : {result.probability:.4f}  (P=fake)")
    print(f"  │  Confidence  : {result.confidence:.4f}")
    print(f"  └───────────────────────────────────────\n")


def cmd_prepare(args):
    from utils.preprocessing import build_manifest, validate_dataset

    logger.info(f"Validating dataset at {args.data_dir}...")
    report = validate_dataset(args.data_dir)
    print(json.dumps(report, indent=2))

    logger.info("Building train/val/test manifests...")
    splits = build_manifest(
        args.data_dir,
        val_split=args.val_split,
        test_split=args.test_split,
    )
    for split, data in splits.items():
        print(f"  {split}: {len(data)} samples")


def cmd_serve(args):
    import uvicorn
    logger.info(f"Starting API server on port {args.port}...")
    import os
    port = int(os.environ.get("PORT", args.port))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=args.dev)


# ─────────────────────────────────────────
#  CLI Parser
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Deepfake Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--data_dir",   required=True)
    p_train.add_argument("--epochs",     type=int,   default=30)
    p_train.add_argument("--batch_size", type=int,   default=32)
    p_train.add_argument("--lr",         type=float, default=1e-4)
    p_train.add_argument("--save_dir",   default="checkpoints")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate on test set")
    p_eval.add_argument("--data_dir",    required=True)
    p_eval.add_argument("--model",       default="checkpoints/best_model.pth")
    p_eval.add_argument("--output_dir",  default="results")

    # predict
    p_pred = sub.add_parser("predict", help="Predict on single file")
    p_pred.add_argument("--input",       required=True)
    p_pred.add_argument("--model",       default="checkpoints/best_model.pth")
    p_pred.add_argument("--no_heatmap",  action="store_true")
    p_pred.add_argument("--save_heatmap", default="heatmap.jpg")

    # prepare
    p_prep = sub.add_parser("prepare", help="Prepare dataset manifests")
    p_prep.add_argument("--data_dir",    required=True)
    p_prep.add_argument("--val_split",   type=float, default=0.15)
    p_prep.add_argument("--test_split",  type=float, default=0.10)

    # serve
    p_srv = sub.add_parser("serve", help="Start REST API server")
    p_srv.add_argument("--port",  type=int, default=8000)
    p_srv.add_argument("--dev",   action="store_true")

    args = parser.parse_args()

    dispatch = {
        "train":   cmd_train,
        "eval":    cmd_eval,
        "predict": cmd_predict,
        "prepare": cmd_prepare,
        "serve":   cmd_serve,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(1)

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
