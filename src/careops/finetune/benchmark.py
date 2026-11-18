"""Compare prompting and decoder fine-tuning on the public development fixture split."""

import argparse
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch

from careops.corpus import DEV_CLAIMS, digest, file_digest, read_jsonl
from careops.embedding import fetch_model, load_model_spec
from careops.finetune.fixtures import LABELS, PROMPT, PROMPT_VERSION, examples
from careops.finetune.training import (
    BATCH, EPOCHS, INPUT_TOKENS, LEARNING_RATE, MODEL_DIR, OUTPUT_TOKENS, SPEC, TRAIN_SEED,
    evaluate, load_adapter, load_base, predict, rank_labels, tune,
)


def score_forms(tokenizer, model, holdout):
    report = {}
    for name, alternate in (("record", False), ("paraphrase", True)):
        outputs = predict(tokenizer, model, holdout, alternate=alternate)
        ranked = rank_labels(tokenizer, model, holdout, alternate=alternate)
        report[name] = {
            "metrics": evaluate(holdout, outputs),
            "outputs": outputs,
            "closed_label_diagnostic": {"metrics": evaluate(holdout, ranked), "outputs": ranked},
        }
    return report


def run(model_dir, adapter_path=None):
    splits = examples(read_jsonl(DEV_CLAIMS))
    tokenizer, model = load_base(model_dir)
    spec = load_model_spec(SPEC)
    holdout = splits["holdout"]
    prompted = score_forms(tokenizer, model, holdout)
    training, adapter = tune(tokenizer, model, splits["train"], splits["validation"])
    tuned = score_forms(tokenizer, model, holdout)
    if adapter_path:
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.write_bytes(adapter)
    report = {
        "scope": "synthetic development claim routing",
        "base": {"repository": spec["repository"], "revision": spec["revision"],
                 "weights_sha256": spec["files"]["model.safetensors"]["sha256"]},
        "input_sha256": file_digest(DEV_CLAIMS),
        "environment": {
            "os": platform.platform(), "python": platform.python_version(), "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "versions": {name: version(name) for name in (
                "torch", "transformers", "tokenizers", "safetensors", "numpy", "matplotlib",
            )},
        },
        "protocol": {
            "prompt_version": PROMPT_VERSION, "prompt_sha256": digest(PROMPT.encode()),
            "train_seed": TRAIN_SEED, "epochs": EPOCHS, "batch": BATCH,
            "learning_rate": LEARNING_RATE, "input_tokens": INPUT_TOKENS,
            "max_new_tokens": OUTPUT_TOKENS,
            "selection": "highest validation macro F1, earliest epoch on ties",
            "holdout_prompt_forms": ["record", "paraphrase"],
            "closed_label_diagnostic": "maximum mean log probability over label tokens and EOS",
            "labels": list(LABELS),
        },
        "splits": {
            name: {"claims": sorted({case.claim_id for case in rows}), "examples": len(rows),
                   "sha256": digest(json.dumps([asdict(case) for case in rows],
                                               sort_keys=True).encode())}
            for name, rows in splits.items()
        },
        "tokenization": {
            "class": type(tokenizer).__name__,
            "embedding_shape": list(model.get_input_embeddings().weight.shape),
            "label_token_ids": {label: tokenizer.encode(label) for label in LABELS},
        },
        "training": training,
        "holdout_ids": [case.sample_id for case in holdout],
        "holdout_labels": [case.label for case in holdout],
        "prompt_only": prompted,
        "fine_tuned": tuned,
    }
    return report


def plot(report, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = report["training"]["history"]
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    fig.suptitle("FLAN-T5 synthetic claim routing", fontsize=17, weight="bold")
    axes[0].plot(epochs, [row["mean_train_loss"] for row in history], marker="o", color="#11695c")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Token cross-entropy")
    axes[1].plot(
        epochs, [row["validation_macro_f1"] for row in history], marker="o", color="#11695c",
    )
    axes[1].axvline(report["training"]["selected_epoch"], color="#b45040", linestyle="--",
                   label="Selected epoch")
    axes[1].set(title="Validation", xlabel="Epoch", ylabel="Macro F1", ylim=(0, 1.05))
    axes[1].legend(fontsize=8)
    for offset, method, label, color in (
        (-0.17, "prompt_only", "Prompt only", "#697f94"),
        (0.17, "fine_tuned", "Fine-tuned", "#11695c"),
    ):
        values = [report[method][form]["metrics"]["macro_f1"]
                  for form in ("record", "paraphrase")]
        axes[2].bar([offset, 1 + offset], values, width=0.34, color=color, label=label)
    axes[2].set(title="Development holdout", ylabel="Macro F1", ylim=(0, 1.05))
    axes[2].set_xticks([0, 1], ["Record", "Paraphrase"])
    axes[2].legend(fontsize=8)
    fig.savefig(target, dpi=145)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    fetch = actions.add_parser("fetch-model")
    train = actions.add_parser("train")
    adapter = actions.add_parser("score-adapter")
    for command in (fetch, train, adapter):
        command.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    train.add_argument("--adapter", type=Path)
    train.add_argument("--figure", type=Path)
    adapter.add_argument("adapter", type=Path)
    args = parser.parse_args()

    if args.action == "fetch-model":
        fetch_model(args.model_dir, SPEC)
        return
    if args.action == "train":
        report = run(args.model_dir, args.adapter)
        if args.figure:
            plot(report, args.figure)
    else:
        tokenizer, model = load_base(args.model_dir)
        load_adapter(model, args.adapter)
        report = score_forms(tokenizer, model, examples(read_jsonl(DEV_CLAIMS))["holdout"])
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
