"""Run the data and model lab using the public synthetic development claims."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

from careops.corpus import DEV_CLAIMS, file_digest, read_jsonl
from careops.model_lab.fixtures import SEED, make_scans
from careops.model_lab.tabular import clustering, exploratory, forecast, predict_amount
from careops.model_lab.vision import compare

DISPLAY_NAMES = {
    "majority": "Majority", "sklearn_logistic": "Logistic",
    "tensorflow_cnn": "TensorFlow CNN", "pytorch_cnn": "PyTorch CNN",
    "huggingface_vit": "HF ViT",
}


def run():
    rows = read_jsonl(DEV_CLAIMS)
    if any(row.get("synthetic") is not True for row in rows):
        raise ValueError("The lab accepts development-only synthetic claims")
    scans = make_scans(rows)
    return ({
        "scope": "synthetic development fixtures only",
        "seed": SEED,
        "input_sha256": file_digest(DEV_CLAIMS),
        "environment": {"os": platform.platform(), "python": platform.python_version(),
                        "device": "cpu",
                        "versions": {name: version(name) for name in (
                            "numpy", "scipy", "pillow", "scikit-learn", "torch", "tensorflow", "keras",
                            "transformers", "matplotlib",
                        )}},
        "exploration": exploratory(rows),
        "clustering": clustering(rows),
        "amount_prediction": predict_amount(rows),
        "workload_forecast": forecast(rows),
        "document_vision": compare(scans),
    }, scans["holdout"])


def plot(report, holdout, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vision = report["document_vision"]
    labels = vision["labels"]
    models = vision["models"]
    forecast_rows = report["workload_forecast"]["predictions"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    fig.suptitle("CareOps synthetic fixture experiments", fontsize=17, weight="bold")

    training = axes[0, 0]
    for name, row in models.items():
        if "history" in row:
            training.plot([item["epoch"] for item in row["history"]],
                          [item["validation_macro_f1"] for item in row["history"]],
                          label=DISPLAY_NAMES[name])
    training.set(title="Document model validation", xlabel="Epoch", ylabel="Macro F1", ylim=(0, 1))
    training.legend(fontsize=8)

    results = axes[0, 1]
    names = [DISPLAY_NAMES[name] for name in models]
    scores = [row["holdout_metrics"]["macro_f1"] for row in models.values()]
    results.barh(names, scores, color="#268477")
    results.invert_yaxis()
    results.set(title="Development holdout", xlabel="Macro F1", xlim=(0, 1))

    workload = axes[0, 2]
    for name, style in (("arrivals", "-"), ("seasonal_naive", ":"), ("ridge", "--")):
        workload.plot([row[name] for row in forecast_rows], style, label=name.replace("_", " "))
    workload.set(title="Simulated arrivals", xlabel="Holdout day", ylabel="Claims")
    workload.legend(fontsize=8)

    for index, label in enumerate(labels):
        candidates = [i for i, item in enumerate(holdout.labels) if item == index]
        selected = max(candidates, key=lambda i: sum(
            row["holdout_predictions"][i] != index for name, row in models.items()
            if name != "majority"
        ))
        ax = axes[1, index]
        ax.imshow(holdout.pixels[selected], cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        vit_guess = labels[models["huggingface_vit"]["holdout_predictions"][selected]]
        ax.set_title(f"Actual {label} · ViT predicted {vit_guess}", fontsize=10)
        ax.set_xlabel(holdout.ids[selected].split(":")[0], fontsize=9)

    fig.savefig(output, dpi=145)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    report, holdout = run()
    if args.figure:
        plot(report, holdout, args.figure)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
