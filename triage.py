"""PyTorch-backed triage pipeline for TorchReview Copilot."""

from __future__ import annotations

import ast
import hashlib
import os
import re
import time
from functools import lru_cache
from typing import List, Sequence

import torch
import torch.nn.functional as F

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    AutoModel = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

try:
    from .triage_catalog import build_examples, build_prototypes
    from .triage_models import (
        IssueLabel,
        PrototypeMatch,
        TriageExample,
        TriagePrototype,
        TriageResult,
        TriageSignal,
    )
except ImportError:
    from triage_catalog import build_examples, build_prototypes
    from triage_models import (
        IssueLabel,
        PrototypeMatch,
        TriageExample,
        TriagePrototype,
        TriageResult,
        TriageSignal,
    )


MODEL_ID = os.getenv("TRIAGE_MODEL_ID", "huggingface/CodeBERTa-small-v1")
MODEL_MAX_LENGTH = int(os.getenv("TRIAGE_MODEL_MAX_LENGTH", "256"))
LABELS: tuple[IssueLabel, ...] = ("syntax", "logic", "performance")


class _LoopDepthVisitor(ast.NodeVisitor):
    """Track the maximum loop nesting depth in a code snippet."""

    def __init__(self) -> None:
        self.depth = 0
        self.max_depth = 0

    def _visit_loop(self, node: ast.AST) -> None:
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        self.generic_visit(node)
        self.depth -= 1

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        self._visit_loop(node)

    def visit_While(self, node: ast.While) -> None:  # noqa: N802
        self._visit_loop(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:  # noqa: N802
        self._visit_loop(node)


class HashingEmbeddingBackend:
    """Deterministic torch-native fallback when pretrained weights are unavailable."""

    def __init__(self, dimensions: int = 96) -> None:
        self.dimensions = dimensions
        self.model_id = "hashed-token-fallback"
        self.backend_name = "hashed-token-fallback"
        self.notes = ["Using hashed torch embeddings because pretrained weights are unavailable."]

    def embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        rows = torch.zeros((len(texts), self.dimensions), dtype=torch.float32)
        for row_index, text in enumerate(texts):
            tokens = re.findall(r"[A-Za-z_]+|\d+|==|!=|<=|>=|\S", text.lower())[:512]
            if not tokens:
                rows[row_index, 0] = 1.0
                continue
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                bucket = int(digest[:8], 16) % self.dimensions
                sign = -1.0 if int(digest[8:10], 16) % 2 else 1.0
                rows[row_index, bucket] += sign
        return F.normalize(rows + 1e-6, dim=1)


class TransformersEmbeddingBackend:
    """Mean-pool CodeBERTa embeddings via torch + transformers."""

    def __init__(self, model_id: str = MODEL_ID, force_fallback: bool = False) -> None:
        self.model_id = model_id
        self.force_fallback = force_fallback
        self.backend_name = model_id
        self.notes: List[str] = []
        self._fallback = HashingEmbeddingBackend()
        self._tokenizer = None
        self._model = None
        self._load_error = ""
        if force_fallback:
            self.backend_name = self._fallback.backend_name
            self.notes = list(self._fallback.notes)

    def _ensure_loaded(self) -> None:
        if self.force_fallback or self._model is not None or self._load_error:
            return
        if AutoTokenizer is None or AutoModel is None:
            self._load_error = "transformers is not installed."
        else:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModel.from_pretrained(self.model_id)
                self._model.eval()
                self.notes.append(f"Loaded pretrained encoder `{self.model_id}` for inference.")
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"

        if self._load_error:
            self.backend_name = self._fallback.backend_name
            self.notes = list(self._fallback.notes) + [f"Pretrained load failed: {self._load_error}"]

    def embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        self._ensure_loaded()
        if self._model is None or self._tokenizer is None:
            return self._fallback.embed_texts(texts)

        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=MODEL_MAX_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self._model(**encoded)
            hidden_state = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return F.normalize(pooled, dim=1)


def _sanitize_text(value: str) -> str:
    text = (value or "").strip()
    return text[:4000]


def _safe_softmax(scores: dict[IssueLabel, float]) -> dict[str, float]:
    tensor = torch.tensor([scores[label] for label in LABELS], dtype=torch.float32)
    probabilities = torch.softmax(tensor * 4.0, dim=0)
    return {label: round(float(probabilities[index]), 4) for index, label in enumerate(LABELS)}


def _loop_depth(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    visitor = _LoopDepthVisitor()
    visitor.visit(tree)
    return visitor.max_depth


def _repair_risk(label: IssueLabel, confidence: float, signal_count: int) -> str:
    base = {"syntax": 0.25, "logic": 0.55, "performance": 0.7}[label]
    if confidence < 0.55:
        base += 0.12
    if signal_count >= 4:
        base += 0.08
    if base < 0.4:
        return "low"
    if base < 0.72:
        return "medium"
    return "high"


class CodeTriageEngine:
    """Combine static signals with PyTorch embeddings to classify code issues."""

    def __init__(
        self,
        *,
        backend: TransformersEmbeddingBackend | HashingEmbeddingBackend | None = None,
        prototypes: Sequence[TriagePrototype] | None = None,
        examples: Sequence[TriageExample] | None = None,
    ) -> None:
        self.backend = backend or TransformersEmbeddingBackend()
        self.prototypes = list(prototypes or build_prototypes())
        self.examples = list(examples or build_examples())
        self._prototype_matrix: torch.Tensor | None = None

    def example_map(self) -> dict[str, TriageExample]:
        """Return UI examples keyed by task id."""

        return {example.key: example for example in self.examples}

    def _build_document(self, code: str, traceback_text: str) -> str:
        trace = _sanitize_text(traceback_text) or "No traceback supplied."
        snippet = _sanitize_text(code) or "# No code supplied."
        return f"Candidate code:\n{snippet}\n\nObserved failure:\n{trace}\n"

    def _prototype_embeddings(self) -> torch.Tensor:
        if self._prototype_matrix is None:
            reference_texts = [prototype.reference_text for prototype in self.prototypes]
            self._prototype_matrix = self.backend.embed_texts(reference_texts)
        return self._prototype_matrix

    def _extract_signals(self, code: str, traceback_text: str) -> tuple[list[TriageSignal], dict[IssueLabel, float], list[str]]:
        trace = (traceback_text or "").lower()
        heuristic_scores: dict[IssueLabel, float] = {label: 0.15 for label in LABELS}
        signals: list[TriageSignal] = []
        notes: list[str] = []

        try:
            ast.parse(code)
            signals.append(
                TriageSignal(
                    name="syntax_parse",
                    value="passes",
                    impact="syntax",
                    weight=0.1,
                    evidence="Python AST parsing succeeded.",
                )
            )
            heuristic_scores["logic"] += 0.05
        except SyntaxError as exc:
            evidence = f"{exc.msg} at line {exc.lineno}"
            signals.append(
                TriageSignal(
                    name="syntax_parse",
                    value="fails",
                    impact="syntax",
                    weight=0.95,
                    evidence=evidence,
                )
            )
            heuristic_scores["syntax"] += 0.85
            notes.append(f"Parser failure detected: {evidence}")

        if any(token in trace for token in ("syntaxerror", "indentationerror", "expected ':'")):
            signals.append(
                TriageSignal(
                    name="traceback_keyword",
                    value="syntaxerror",
                    impact="syntax",
                    weight=0.8,
                    evidence="Traceback contains a parser error.",
                )
            )
            heuristic_scores["syntax"] += 0.55

        if any(token in trace for token in ("assertionerror", "expected:", "actual:", "boundary", "missing", "incorrect")):
            signals.append(
                TriageSignal(
                    name="test_failure_signal",
                    value="assertion-style failure",
                    impact="logic",
                    weight=0.7,
                    evidence="Failure text points to behavioral mismatch instead of parser issues.",
                )
            )
            heuristic_scores["logic"] += 0.55

        if any(token in trace for token in ("timeout", "benchmark", "slow", "latency", "performance", "profiler")):
            signals.append(
                TriageSignal(
                    name="performance_trace",
                    value="latency regression",
                    impact="performance",
                    weight=0.85,
                    evidence="Traceback mentions benchmark or latency pressure.",
                )
            )
            heuristic_scores["performance"] += 0.7

        loop_depth = _loop_depth(code)
        if loop_depth >= 2:
            signals.append(
                TriageSignal(
                    name="loop_depth",
                    value=str(loop_depth),
                    impact="performance",
                    weight=0.65,
                    evidence="Nested iteration increases runtime risk on larger fixtures.",
                )
            )
            heuristic_scores["performance"] += 0.35

        if "Counter(" in code or "defaultdict(" in code or "set(" in code:
            heuristic_scores["performance"] += 0.05

        if "return sessions" in code and "sessions.append" not in code:
            signals.append(
                TriageSignal(
                    name="state_update_gap",
                    value="possible missing final append",
                    impact="logic",
                    weight=0.45,
                    evidence="A collection is returned without an obvious final state flush.",
                )
            )
            heuristic_scores["logic"] += 0.18

        return signals, heuristic_scores, notes

    def _nearest_match(self, embedding: torch.Tensor) -> tuple[TriagePrototype, float, dict[str, float]]:
        similarities = torch.matmul(embedding, self._prototype_embeddings().T)[0]
        indexed_scores = {
            self.prototypes[index].task_id: round(float((similarities[index] + 1.0) / 2.0), 4)
            for index in range(len(self.prototypes))
        }
        best_index = int(torch.argmax(similarities).item())
        best_prototype = self.prototypes[best_index]
        best_similarity = float((similarities[best_index] + 1.0) / 2.0)
        return best_prototype, best_similarity, indexed_scores

    def _repair_plan(self, label: IssueLabel, matched: TriagePrototype) -> list[str]:
        plans = {
            "syntax": [
                "Patch the parser break first: missing colon, bracket, or indentation before changing logic.",
                f"Realign the implementation with the known-good pattern from `{matched.title}`.",
                "Re-run the visible checks once the file compiles, then verify hidden edge cases.",
            ],
            "logic": [
                "Reproduce the failing assertion with the smallest public example and inspect state transitions.",
                f"Compare boundary handling against the known issue pattern `{matched.title}`.",
                "Patch the final state update or branch condition, then rerun correctness checks before submission.",
            ],
            "performance": [
                "Profile the hot path and isolate repeated full-list scans or nested loops.",
                f"Refactor toward counting or indexing strategies similar to `{matched.title}`.",
                "Benchmark the new implementation on a production-like fixture and confirm output stability.",
            ],
        }
        return plans[label]

    def triage(self, code: str, traceback_text: str = "") -> TriageResult:
        """Run the full triage pipeline on code plus optional failure context."""

        started = time.perf_counter()
        document = self._build_document(code, traceback_text)
        signals, heuristic_scores, notes = self._extract_signals(code, traceback_text)

        candidate_embedding = self.backend.embed_texts([document])
        matched, matched_similarity, prototype_scores = self._nearest_match(candidate_embedding)

        label_similarity = {label: 0.18 for label in LABELS}
        for prototype in self.prototypes:
            label_similarity[prototype.label] = max(
                label_similarity[prototype.label],
                prototype_scores[prototype.task_id],
            )

        combined_scores = {
            label: 0.72 * label_similarity[label] + 0.28 * heuristic_scores[label]
            for label in LABELS
        }
        confidence_scores = _safe_softmax(combined_scores)
        issue_label = max(LABELS, key=lambda label: confidence_scores[label])
        top_confidence = confidence_scores[issue_label]

        top_signal = signals[0].evidence if signals else "Model similarity dominated the decision."
        summary = (
            f"Detected a {issue_label} issue with {top_confidence:.0%} confidence. "
            f"The closest known failure pattern is `{matched.title}`, which indicates {matched.summary.lower()}"
        )
        suggested_next_action = {
            "syntax": "Fix the parser error first, then rerun validation before changing behavior.",
            "logic": "Step through the smallest failing case and confirm the final branch/update behavior.",
            "performance": "Replace repeated full-list scans with a linear-time aggregation strategy, then benchmark it.",
        }[issue_label]

        return TriageResult(
            issue_label=issue_label,
            confidence_scores=confidence_scores,
            repair_risk=_repair_risk(issue_label, top_confidence, len(signals)),
            summary=summary,
            matched_pattern=PrototypeMatch(
                task_id=matched.task_id,
                title=matched.title,
                label=matched.label,
                similarity=round(matched_similarity, 4),
                summary=matched.summary,
                rationale=top_signal,
            ),
            repair_plan=self._repair_plan(issue_label, matched),
            suggested_next_action=suggested_next_action,
            extracted_signals=signals,
            model_backend=self.backend.backend_name,
            model_id=self.backend.model_id,
            inference_notes=list(self.backend.notes) + notes,
            analysis_time_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )


@lru_cache(maxsize=1)
def get_default_engine() -> CodeTriageEngine:
    """Return a cached triage engine for the running process."""

    return CodeTriageEngine()
