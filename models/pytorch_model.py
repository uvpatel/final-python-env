"""PyTorch + transformers model wrapper for multi-domain code scoring."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    AutoModel = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


DOMAIN_PROTOTYPES: Dict[str, List[str]] = {
    "dsa": [
        "Binary search, hashmap optimization, recursion, dynamic programming, arrays, trees, graphs, stack, queue, complexity.",
        "Competitive programming algorithm with loops, memoization, prefix sums, and asymptotic analysis.",
    ],
    "data_science": [
        "Pandas dataframe transformation, numpy vectorization, feature leakage, train test split, iterrows misuse.",
        "Data cleaning pipeline using pandas, numpy, aggregation, joins, and vectorized operations.",
    ],
    "ml_dl": [
        "PyTorch model, training loop, optimizer, backward pass, eval mode, no_grad, loss function, dataloader.",
        "Machine learning inference and training code with torch, sklearn, tensors, gradients, and model checkpoints.",
    ],
    "web": [
        "FastAPI endpoint, request validation, Pydantic models, async routes, API security, backend service design.",
        "REST API backend with routers, dependency injection, input validation, serialization, and error handling.",
    ],
    "general": [
        "General Python utility code with readable structure, typing, tests, and maintainable abstractions.",
    ],
}

QUALITY_ANCHORS: Dict[str, List[str]] = {
    "high": [
        "Readable typed Python code with validation, efficient algorithms, vectorized operations, safe inference, and clean API boundaries.",
        "Production-ready code with small functions, docstrings, low complexity, and clear error handling.",
    ],
    "low": [
        "Brute-force nested loops, missing validation, unsafe input handling, missing eval mode, missing no_grad, and code smells.",
        "Hard to maintain code with high complexity, repeated scans, mutable side effects, and unclear structure.",
    ],
}


class _HashEmbeddingBackend:
    """Torch-native fallback when pretrained weights cannot be loaded."""

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions
        self.model_id = "hashed-token-fallback"
        self.backend_name = "hashed-token-fallback"
        self.notes = ["Using hashed embeddings because pretrained transformer weights are unavailable."]

    def embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        matrix = torch.zeros((len(texts), self.dimensions), dtype=torch.float32)
        for row_index, text in enumerate(texts):
            tokens = text.lower().split()[:512]
            if not tokens:
                matrix[row_index, 0] = 1.0
                continue
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                bucket = int(digest[:8], 16) % self.dimensions
                sign = -1.0 if int(digest[8:10], 16) % 2 else 1.0
                matrix[row_index, bucket] += sign
        return F.normalize(matrix + 1e-6, dim=1)


class PyTorchCodeAnalyzerModel:
    """Score code using pretrained transformer embeddings plus prototype similarity."""

    def __init__(self, model_id: str = "huggingface/CodeBERTa-small-v1") -> None:
        self.model_id = model_id
        self.backend_name = model_id
        self.notes: List[str] = []
        self._tokenizer = None
        self._model = None
        self._fallback = _HashEmbeddingBackend()
        self._prototype_cache: Dict[str, torch.Tensor] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None or self.notes:
            return
        if AutoTokenizer is None or AutoModel is None:
            self.backend_name = self._fallback.backend_name
            self.notes = list(self._fallback.notes)
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id)
            self._model.eval()
            self.notes.append(f"Loaded pretrained encoder `{self.model_id}`.")
        except Exception as exc:
            self.backend_name = self._fallback.backend_name
            self.notes = list(self._fallback.notes) + [f"Pretrained load failed: {type(exc).__name__}: {exc}"]

    def _embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        self._ensure_loaded()
        if self._model is None or self._tokenizer is None:
            return self._fallback.embed_texts(texts)
        encoded = self._tokenizer(list(texts), padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return F.normalize(pooled, dim=1)

    def _prototype_matrix(self, bucket: str, texts: Sequence[str]) -> torch.Tensor:
        if bucket not in self._prototype_cache:
            self._prototype_cache[bucket] = self._embed_texts(texts)
        return self._prototype_cache[bucket]

    def predict(self, code: str, context_window: str, static_summary: Dict[str, object]) -> Dict[str, object]:
        """Predict domain probabilities and a model quality score."""

        document = (
            f"Code:\n{code.strip()[:4000]}\n\n"
            f"Context:\n{context_window.strip()[:1000]}\n\n"
            f"Static hints:\n{static_summary}\n"
        )
        candidate = self._embed_texts([document])

        domain_scores: Dict[str, float] = {}
        for domain, texts in DOMAIN_PROTOTYPES.items():
            matrix = self._prototype_matrix(f"domain:{domain}", texts)
            similarity = torch.matmul(candidate, matrix.T).max().item()
            domain_scores[domain] = round((similarity + 1.0) / 2.0, 4)

        high_matrix = self._prototype_matrix("quality:high", QUALITY_ANCHORS["high"])
        low_matrix = self._prototype_matrix("quality:low", QUALITY_ANCHORS["low"])
        high_similarity = torch.matmul(candidate, high_matrix.T).max().item()
        low_similarity = torch.matmul(candidate, low_matrix.T).max().item()
        ml_quality_score = torch.sigmoid(torch.tensor((high_similarity - low_similarity) * 4.0)).item()

        return {
            "domain_scores": domain_scores,
            "ml_quality_score": round(float(ml_quality_score), 4),
            "backend_name": self.backend_name,
            "model_id": self.model_id,
            "notes": list(self.notes),
        }
