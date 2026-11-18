"""Offline multilingual E5 embeddings and an explicit, checksum-checked download."""

import json
from hashlib import file_digest
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

MODEL_SPEC = Path("data/manifests/embedding.json")
MODEL_DIRECTORY = Path("models/multilingual-e5-small")


def load_model_spec(spec_path=MODEL_SPEC):
    return json.loads(spec_path.read_text(encoding="utf-8"))


def check_model_file(path, info):
    if path.stat().st_size != info["bytes"]:
        raise ValueError(f"{path.name}: unexpected model file size")
    with path.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != info["sha256"]:
            raise ValueError(f"{path.name}: model SHA-256 mismatch")


def fetch_model(directory, spec_path=MODEL_SPEC):
    spec = load_model_spec(spec_path)
    base = f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}"
    directory.mkdir(parents=True, exist_ok=True)
    for name, info in spec["files"].items():
        path = directory / name
        if path.exists():
            check_model_file(path, info)
            continue
        with NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with urlopen(f"{base}/{name}", timeout=60) as response, temporary.open("wb") as stream:
                remaining = info["bytes"]
                while remaining:
                    chunk = response.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        break
                    stream.write(chunk)
                    remaining -= len(chunk)
                if remaining or response.read(1):
                    raise ValueError(f"{name}: unexpected download size")
            check_model_file(temporary, info)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


class E5Encoder:
    def __init__(self, directory):
        import torch
        from transformers import AutoModel, AutoTokenizer

        for name, info in load_model_spec()["files"].items():
            check_model_file(directory / name, info)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            directory, local_files_only=True, use_safetensors=True
        ).eval()

    def _encode(self, texts, prefix):
        batches = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), 8):
                batch = [f"{prefix}: {text}" for text in texts[start:start + 8]]
                inputs = self.tokenizer(
                    batch, max_length=512, padding=True, truncation=True, return_tensors="pt"
                )
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).bool()
                pooled = hidden.masked_fill(~mask, 0.0).sum(1) / mask.sum(1)
                batches.append(self.torch.nn.functional.normalize(pooled, dim=1))
        return self.torch.cat(batches).numpy()

    def passages(self, texts):
        return self._encode(texts, "passage")

    def query(self, text):
        return self._encode([text], "query")[0]
