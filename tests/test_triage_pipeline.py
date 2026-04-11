from __future__ import annotations

from fastapi.testclient import TestClient

from triage import CodeTriageEngine, HashingEmbeddingBackend
from triage_catalog import build_examples


def test_hashing_backend_returns_normalized_embeddings() -> None:
    backend = HashingEmbeddingBackend(dimensions=32)
    embeddings = backend.embed_texts(["def foo():\n    return 1", "for x in items:\n    pass"])

    assert embeddings.shape == (2, 32)
    for row in embeddings:
        assert round(float(row.norm().item()), 5) == 1.0


def test_examples_map_to_expected_labels_with_fallback_backend() -> None:
    examples = build_examples()
    engine = CodeTriageEngine(backend=HashingEmbeddingBackend())

    for example in examples:
        result = engine.triage(example.code, example.traceback_text)
        assert result.issue_label == example.label


def test_syntax_example_exposes_parser_signal() -> None:
    example = next(item for item in build_examples() if item.label == "syntax")
    engine = CodeTriageEngine(backend=HashingEmbeddingBackend())

    result = engine.triage(example.code, example.traceback_text)

    assert any(signal.name == "syntax_parse" and signal.value == "fails" for signal in result.extracted_signals)
    assert result.matched_pattern.task_id == example.task_id


def test_composed_app_preserves_health_route() -> None:
    from server.app import build_application

    client = TestClient(build_application())
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
