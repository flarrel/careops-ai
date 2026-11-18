"""Explore fixture patterns, predict fixture amounts, and forecast simulated workload."""

from collections import Counter
from datetime import date, timedelta
from itertools import combinations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, mean_absolute_error, silhouette_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from careops.agent_tools import claim_issues
from careops.model_lab.fixtures import SEED

DAYS = 84
FORECAST_START = 63


def exploratory(rows):
    amounts = np.asarray([row["amount_idr"] for row in rows])
    template_counts = Counter(row["template"] for row in rows)
    attachment_counts = Counter(item for row in rows for item in set(row["attachments"]))
    bins = Counter()
    pairs = Counter()
    for row in rows:
        items = {
            f"service={row['service_type']}",
            f"checklist={'flag' if claim_issues(row, None) else 'clear'}",
            *(f"attachment={item}" for item in row["attachments"]),
        }
        bins.update(items)
        pairs.update(combinations(sorted(items), 2))
    associations = [
        {"items": list(pair), "count": count,
         "support": round(count / len(rows), 4),
         "lift": round(count * len(rows) / (bins[pair[0]] * bins[pair[1]]), 4)}
        for pair, count in pairs.items() if count >= 3
    ]
    associations.sort(key=lambda pair: (-pair["lift"], -pair["count"], pair["items"]))
    return {
        "claims": len(rows),
        "amount_idr": {"min": int(amounts.min()), "median": float(np.median(amounts)),
                       "max": int(amounts.max())},
        "templates": dict(template_counts),
        "attachments": dict(attachment_counts),
        "checklist_flagged": bins["checklist=flag"],
        "top_cooccurrences": associations[:8],
    }


def clustering(rows):
    source = np.asarray([
        [np.log1p(row["amount_idr"]), len(row["attachments"])] for row in rows
    ])
    scaled = StandardScaler().fit_transform(source)
    search = []
    for k in (2, 3, 4):
        model = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(scaled)
        search.append({
            "k": k,
            "silhouette": round(float(silhouette_score(scaled, model.labels_)), 4),
            "inertia": round(float(model.inertia_), 4),
            "ari_vs_fixture_templates": round(float(adjusted_rand_score(
                [row["template"] for row in rows], model.labels_,
            )), 4),
            "groups": [
                {"cluster": i, "count": int(sum(model.labels_ == i)),
                 "amount_median_idr": float(np.median([
                     row["amount_idr"] for row, label in zip(rows, model.labels_, strict=True)
                     if label == i
                 ]))}
                for i in range(k)
            ],
        })
    return {"features": ["log_amount", "attachment_count"],
            "best_k_by_silhouette": max(search, key=lambda candidate: candidate["silhouette"])["k"],
            "candidates": search}


def predict_amount(rows):
    # Amount and fixture template are kept outside the feature matrix.
    inputs = np.array([[row["service_type"], len(row["attachments"])] for row in rows],
                      dtype=object)
    target = np.asarray([row["amount_idr"] for row in rows], dtype=float)
    groups = np.array([row["template"] for row in rows])
    outputs = {}
    predictions = {"mean": np.empty(len(rows)), "ridge": np.empty(len(rows))}
    folds = []
    for train, test in LeaveOneGroupOut().split(inputs, target, groups):
        transform = ColumnTransformer([
            ("service", OneHotEncoder(handle_unknown="ignore"), [0]),
            ("attachments", StandardScaler(), [1]),
        ])
        models = {"mean": DummyRegressor(strategy="mean"),
                  "ridge": make_pipeline(transform, Ridge(alpha=1.0))}
        for name, model in models.items():
            model.fit(inputs[train], target[train])
            predictions[name][test] = model.predict(inputs[test])
        folds.append({
            "template": str(groups[test[0]]),
            "n": len(test),
            "mae_idr": {name: round(float(mean_absolute_error(target[test], pred[test])), 2)
                        for name, pred in predictions.items()},
        })
    for name, pred in predictions.items():
        outputs[name] = round(float(mean_absolute_error(target, pred)), 2)
    return {"features": ["service_type", "attachment_count"], "folds": folds,
            "mae_idr": outputs}


def simulate_arrivals(rows):
    rng = np.random.default_rng(SEED + 1)
    pattern = (13, 15, 14, 12, 11, 6, 4)
    flags = [bool(claim_issues(row, None)) for row in rows]
    volume, flagged = [], []
    for day in range(DAYS):
        rate = pattern[day % 7] + 0.035 * day
        count = int(rng.poisson(rate))
        volume.append(count)
        flagged.append(int(np.count_nonzero(rng.choice(flags, size=count))))
    return volume, flagged


def lag_features(history, day):
    return [history[-1], history[-7], float(np.mean(history[-7:])),
            *[int(day % 7 == weekday) for weekday in range(7)]]


def forecast(rows):
    volume, flagged = simulate_arrivals(rows)
    model = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
    train_days = range(7, FORECAST_START)
    model.fit([lag_features(volume[:d], d) for d in train_days], volume[7:FORECAST_START])
    times = []
    for day in range(FORECAST_START, DAYS):
        history = volume[:day]
        times.append({
            "date": (date(2025, 1, 6) + timedelta(days=day)).isoformat(),
            "arrivals": volume[day], "flagged": flagged[day],
            "seasonal_naive": history[-7],
            "ridge": round(max(0.0, float(model.predict([lag_features(history, day)])[0])), 4),
        })
    truth = [row["arrivals"] for row in times]
    return {
        "simulated_days": DAYS, "train_before_day": FORECAST_START, "predictions": times,
        "mae": {method: round(float(mean_absolute_error(
            truth, [row[method] for row in times],
        )), 4) for method in ("seasonal_naive", "ridge")},
    }
