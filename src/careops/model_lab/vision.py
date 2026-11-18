"""Train classical and neural models on the same rendered document scans."""

from hashlib import sha256

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from careops.model_lab.fixtures import CLASSES, IMAGE_SIZE, SEED

EPOCHS = 14
BATCH = 16
LABELS = list(range(len(CLASSES)))


def score(expected, predictions):
    return {
        "accuracy": round(float(accuracy_score(expected, predictions)), 6),
        "macro_f1": round(float(f1_score(
            expected, predictions, average="macro", labels=LABELS, zero_division=0,
        )), 6),
        "confusion": confusion_matrix(expected, predictions, labels=LABELS).tolist(),
    }


def gray_input(scan):
    return (255 - scan.pixels.astype(np.float32)) / 255.0


def document_features(scan):
    ink = gray_input(scan)
    horizontal = np.abs(np.diff(ink, axis=2, prepend=ink[:, :, :1]))
    vertical = np.abs(np.diff(ink, axis=1, prepend=ink[:, :1, :]))
    return np.concatenate([
        array.reshape(len(ink), 8, 8, 8, 8).mean(axis=(2, 4)).reshape(len(ink), -1)
        for array in (ink, horizontal, vertical)
    ], axis=1)


def batches(count):
    rng = np.random.default_rng(SEED)
    for _ in range(EPOCHS):
        order = rng.permutation(count)
        yield [order[start:start + BATCH] for start in range(0, count, BATCH)]


def history_entry(epoch, train_losses, val_loss, validation, val_prediction):
    return {
        "epoch": epoch,
        "train_loss": round(float(np.mean(train_losses)), 6),
        "validation_loss": round(float(val_loss), 6),
        "validation_macro_f1": score(validation.labels, val_prediction)["macro_f1"],
    }


def choice(row):
    return (row["validation_macro_f1"], -row["validation_loss"])


def train_pytorch(train, validation, holdout, *, transformer=False):
    import torch
    from torch import nn

    torch.manual_seed(SEED)
    if transformer:
        from transformers import ViTConfig, ViTForImageClassification

        config = ViTConfig(
            image_size=IMAGE_SIZE, patch_size=8, num_channels=1, hidden_size=48,
            num_hidden_layers=2, num_attention_heads=4, intermediate_size=96,
            num_labels=len(CLASSES),
        )
        model = ViTForImageClassification(config)

        def predict(images):
            return model(pixel_values=images).logits
    else:
        model = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 16 * 16, 24), nn.ReLU(), nn.Linear(24, len(CLASSES)),
        )

        def predict(images):
            return model(images)

    def tensors(split):
        return torch.from_numpy(gray_input(split)[:, None, :, :]), torch.from_numpy(split.labels)

    x, y = tensors(train)
    val_x, val_y = tensors(validation)
    test_x, _ = tensors(holdout)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    best, best_weights = None, None
    for epoch, group in enumerate(batches(len(y)), start=1):
        model.train()
        losses = []
        for indices in group:
            optimizer.zero_grad()
            loss = loss_fn(predict(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.inference_mode():
            logits = predict(val_x)
            row = history_entry(
                epoch, losses, loss_fn(logits, val_y).item(), validation,
                logits.argmax(1).numpy(),
            )
        history.append(row)
        if best is None or choice(row) > choice(best):
            best = row
            best_weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_weights)
    model.eval()
    with torch.inference_mode():
        prediction = predict(test_x).argmax(1).tolist()
    return {"selected_epoch": best["epoch"], "history": history,
            "parameters": sum(value.numel() for value in model.parameters()),
            "holdout_predictions": prediction}


def train_tensorflow(train, validation, holdout):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    layers = tf.keras.layers
    with tf.device("/CPU:0"):
        model = tf.keras.Sequential([
            layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
            layers.Conv2D(8, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
            layers.Conv2D(16, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
            layers.Flatten(), layers.Dense(24, activation="relu"), layers.Dense(len(CLASSES)),
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        x = gray_input(train)[..., None]
        val_x = gray_input(validation)[..., None]
        history = []
        best, best_weights = None, None
        for epoch, group in enumerate(batches(len(train.labels)), start=1):
            losses = []
            for indices in group:
                with tf.GradientTape() as tape:
                    logits = model(x[indices], training=True)
                    loss = loss_fn(train.labels[indices], logits)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables, strict=True))
                losses.append(float(loss.numpy()))
            logits = model(val_x, training=False)
            row = history_entry(
                epoch, losses, loss_fn(validation.labels, logits).numpy(), validation,
                tf.argmax(logits, axis=1).numpy(),
            )
            history.append(row)
            if best is None or choice(row) > choice(best):
                best, best_weights = row, model.get_weights()
        model.set_weights(best_weights)
        prediction = tf.argmax(model(gray_input(holdout)[..., None], training=False), axis=1)
    return {"selected_epoch": best["epoch"], "history": history,
            "parameters": model.count_params(),
            "holdout_predictions": prediction.numpy().tolist()}


def compare(scans):
    train, validation, holdout = (scans[key] for key in ("train", "validation", "holdout"))
    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "sklearn_logistic": make_pipeline(
            StandardScaler(), LogisticRegression(solver="newton-cg", max_iter=1000),
        ),
    }
    x, test_x = document_features(train), document_features(holdout)
    measured = {
        name: {"holdout_predictions": model.fit(x, train.labels).predict(test_x).tolist()}
        for name, model in models.items()
    }
    measured.update({
        "tensorflow_cnn": train_tensorflow(train, validation, holdout),
        "pytorch_cnn": train_pytorch(train, validation, holdout),
        "huggingface_vit": train_pytorch(train, validation, holdout, transformer=True),
    })
    for row in measured.values():
        row["holdout_metrics"] = score(holdout.labels, row["holdout_predictions"])
    return {
        "labels": list(CLASSES),
        "split_sizes": {name: len(split.labels) for name, split in scans.items()},
        "split_claim_ids": {
            name: sorted({scan_id.split(":")[0] for scan_id in split.ids})
            for name, split in scans.items()
        },
        "split_pixel_sha256": {
            name: sha256(split.pixels.tobytes()).hexdigest() for name, split in scans.items()
        },
        "holdout_ids": holdout.ids,
        "holdout_expected": holdout.labels.tolist(),
        "models": measured,
    }
