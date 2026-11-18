"""Fine-tune the final FLAN-T5 decoder block using teacher forcing."""

from hashlib import sha256
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from safetensors.torch import load_file, save
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from careops.embedding import check_model_file, load_model_spec
from careops.finetune.fixtures import LABELS

SPEC = Path("data/manifests/flan-t5-small.json")
MODEL_DIR = Path("models/flan-t5-small")
TRAIN_SEED = 1808
EPOCHS = 8
BATCH = 8
LEARNING_RATE = 0.0006
INPUT_TOKENS = 128
OUTPUT_TOKENS = 12


def load_base(directory):
    for name, info in load_model_spec(SPEC)["files"].items():
        check_model_file(directory / name, info)
    tokenizer = AutoTokenizer.from_pretrained(directory, use_fast=True, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        directory, local_files_only=True, use_safetensors=True,
    )
    return tokenizer, model.eval()


def encode(tokenizer, texts, limit):
    batch = tokenizer(texts, padding=True, return_tensors="pt")
    if batch.input_ids.shape[1] > limit:
        raise ValueError(f"Fixture exceeds its {limit}-token budget")
    return batch


def predict(tokenizer, model, rows, *, alternate=False):
    model.eval()
    predictions = []
    for start in range(0, len(rows), BATCH):
        group = rows[start:start + BATCH]
        tokens = encode(tokenizer, [row.alternate if alternate else row.prompt for row in group],
                        INPUT_TOKENS)
        with torch.inference_mode():
            output = model.generate(**tokens, do_sample=False, num_beams=1,
                                    max_new_tokens=OUTPUT_TOKENS)
        predictions.extend(
            " ".join(text.casefold().split())
            for text in tokenizer.batch_decode(output, skip_special_tokens=True)
        )
    return predictions



def rank_labels(tokenizer, model, rows, *, alternate=False):
    model.eval()
    predictions = []
    label_ids = encode(tokenizer, list(LABELS), OUTPUT_TOKENS).input_ids
    for start in range(0, len(rows), BATCH):
        group = rows[start:start + BATCH]
        texts = [row.alternate if alternate else row.prompt for row in group]
        inputs = encode(tokenizer, texts, INPUT_TOKENS)
        repeated_inputs = {key: value.repeat_interleave(len(LABELS), dim=0)
                           for key, value in inputs.items()}
        targets = label_ids.repeat(len(group), 1)
        mask = targets != tokenizer.pad_token_id
        labels = targets.masked_fill(~mask, -100)
        with torch.inference_mode():
            logits = model(**repeated_inputs, labels=labels).logits
            token_scores = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            means = token_scores.masked_fill(~mask, 0).sum(-1) / mask.sum(-1)
        best = means.reshape(len(group), len(LABELS)).argmax(1)
        predictions.extend(LABELS[index] for index in best.tolist())
    return predictions


def evaluate(rows, predictions):
    grid = [[0] * (len(LABELS) + 1) for _ in LABELS]
    exact = 0
    for row, output in zip(rows, predictions, strict=True):
        expected = LABELS.index(row.label)
        predicted = LABELS.index(output) if output in LABELS else len(LABELS)
        grid[expected][predicted] += 1
        exact += expected == predicted
    scores = []
    for index in range(len(LABELS)):
        hits = grid[index][index]
        denominator = sum(grid[index]) + sum(row[index] for row in grid)
        scores.append(2 * hits / denominator if denominator else 0)
    return {
        "n": len(rows),
        "exact_match": round(exact / len(rows), 6),
        "macro_f1": round(mean(scores), 6),
        "invalid_outputs": sum(row[-1] for row in grid),
        "confusion_with_invalid_column": grid,
    }


def decoder_names(model):
    final_block = f"decoder.block.{len(model.decoder.block) - 1}."
    return {name for name, _ in model.named_parameters()
            if name.startswith((final_block, "decoder.final_layer_norm."))}


def adapter_state(model, names):
    return {name: value.detach().cpu().contiguous().clone()
            for name, value in model.named_parameters() if name in names}


def load_adapter(model, path):
    changed = load_file(path)
    if changed.keys() != decoder_names(model):
        raise ValueError("Adapter keys do not match the base decoder")
    model.load_state_dict(changed, strict=False)


def tune(tokenizer, model, train, validation):
    torch.manual_seed(TRAIN_SEED)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(TRAIN_SEED)
    names = decoder_names(model)
    for name, value in model.named_parameters():
        value.requires_grad_(name in names)
    initial = adapter_state(model, names)
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad], lr=LEARNING_RATE,
    )
    history = []
    best_epoch, best_score, best = 0, -1.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(train))
        losses = []
        for start in range(0, len(order), BATCH):
            rows = [train[i] for i in order[start:start + BATCH]]
            terms = encode(tokenizer, [row.prompt for row in rows], INPUT_TOKENS)
            labels = encode(tokenizer, [row.label for row in rows], OUTPUT_TOKENS).input_ids
            labels[labels == tokenizer.pad_token_id] = -100
            optimizer.zero_grad()
            loss = model(**terms, labels=labels).loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        evaluation = evaluate(validation, predict(tokenizer, model, validation))
        score = evaluation["macro_f1"]
        history.append({"epoch": epoch, "mean_train_loss": round(mean(losses), 6),
                        "validation_macro_f1": score,
                        "validation_exact_match": evaluation["exact_match"]})
        if score > best_score:
            best_epoch, best_score = epoch, score
            best = adapter_state(model, names)

    if all(torch.equal(best[name], initial[name]) for name in names):
        raise RuntimeError("Training did not change a selected decoder parameter")
    model.load_state_dict(best, strict=False)
    adapter = save(best)
    return {
        "history": history,
        "selected_epoch": best_epoch,
        "trained_parameters": sum(tensor.numel() for tensor in best.values()),
        "total_parameters": sum(value.numel() for value in model.parameters()),
        "changed_parameter_tensors": sum(not torch.equal(initial[name], best[name])
                                         for name in names),
        "adapter_sha256": sha256(adapter).hexdigest(),
    }, adapter
