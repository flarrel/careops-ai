# Synthetic fixture model lab

The lab reuses 20 public synthetic development claims. It explores fixture patterns, predicts synthetic amounts, forecasts a simulated review workload, and compares document-scan classifiers. The fixture distributions and rendered defects define the domain of these results.

## Reproduce

Run from the repository root with Python 3.12.

```sh
python3 -m pip install -e '.[models]'
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 \
TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 HF_HUB_OFFLINE=1 \
  python3 -m careops.model_lab.benchmark --figure /tmp/model-lab.png \
  > /tmp/model-lab.json
```

The [machine-readable run](../evals/results/model-lab.json) records the input digest, seed, package versions, split IDs, pixel hashes, training and validation history, holdout predictions, and error metrics. It ran on the CPU of an Apple M3 Pro. The model lab trains from initialization with local fixture data.

## Claim data and simulation

- **Exploration and co-occurrence mining.** The unmodified checklist flagged 10 of 20 records when no simulation amount limit was supplied. The mismatched visit-note attachment appeared in five claims, all flagged. The pattern reflects the authored variants and the checklist rule. The report includes support and lift for frequent pairs.
- **Clustering.** K-means uses standardized log amount and attachment count. Of the three settings tried, three clusters had the highest silhouette of 0.9019. They largely separate the high-amount day-care group and single-attachment examples. Four clusters agreed more closely with authored template labels despite a lower silhouette.
- **Amount prediction.** A train-fold mean and ridge regression predict fixture amount. Ridge uses service type and attachment count. A whole fixture template is held out per fold, with encoding and scaling fitted on the training fold. Mean absolute error was Rp975,833 for the mean and Rp594,407 for ridge. Ridge missed the held-out high-amount day-care template by Rp1.94 million on average.
- **Workload forecast.** A seeded 84-day simulation resamples fixture checklist flags and generates arrivals with a weekday pattern and gradual trend. Ridge uses previous counts and weekday features. The model fits before the final 21 days. Each one-step forecast uses observations through the previous day. Mean absolute error was 5.52 claims for the prior-week baseline and 4.03 for ridge. These figures measure recovery of the simulated pattern.

## Document vision

The lab renders 240 grayscale claim slips from the development fixtures. Paired variants for each claim introduce clean, skewed, and occluded conditions. Twelve claim IDs supply 144 training scans. Four other IDs supply 48 validation scans, and four further IDs supply 48 holdout scans. Each split has all three conditions. All renders from a claim belong to its single split.

Scikit-learn uses logistic regression on pooled ink and horizontal and vertical edge maps. The TensorFlow and PyTorch CNNs share the same layer sizes, data splits, and batch schedule. A small Hugging Face Transformers ViT learns from pixels using a randomly initialized configuration. Neural models run for 14 epochs and select weights using validation macro F1, with validation loss as the tie-break. [TensorFlow GradientTape](https://www.tensorflow.org/api_docs/python/tf/GradientTape), [Hugging Face ViT](https://huggingface.co/docs/transformers/v4.47.1/en/model_doc/vit)

| Model | Holdout accuracy | Holdout macro F1 |
|---|---:|---:|
| Majority | 0.3333 | 0.1667 |
| Scikit-learn logistic regression | 0.9375 | 0.9374 |
| TensorFlow CNN | 0.8542 | 0.8559 |
| PyTorch CNN | 0.9583 | 0.9583 |
| Hugging Face ViT | 0.4375 | 0.3395 |

The PyTorch CNN confused one clean and one occluded scan. The ViT classified all 16 clean scans as skewed and recovered five of 16 occluded scans. The CNN and handcrafted edge features fit this small controlled task more effectively in the measured run. Real scanning defects, source-document OCR, and performance on new document layouts remain outside this evaluation.

[Diagnostic figure](../evals/results/model-lab.png) shows validation curves, the holdout comparison, the forecast, and one illustrative scan per condition. The examples are chosen to show errors where available.
