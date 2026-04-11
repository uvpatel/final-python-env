from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from schemas.request import AnalyzeCodeRequest
from services.analysis_service import AnalysisService


def test_analysis_service_detects_web_code() -> None:
    service = AnalysisService()
    request = AnalyzeCodeRequest(
        code="from fastapi import FastAPI\napp = FastAPI()\n\n@app.get('/health')\ndef health():\n    return {'status': 'ok'}\n",
        domain_hint="auto",
    )

    result = service.analyze(request)

    assert result.detected_domain == "web"
    assert 0.0 <= result.score_breakdown.reward <= 1.0
    assert len(result.improvement_plan) == 3


def test_analysis_service_detects_dsa_code() -> None:
    service = AnalysisService()
    request = AnalyzeCodeRequest(
        code="def has_pair(nums, target):\n    for i in range(len(nums)):\n        for j in range(i + 1, len(nums)):\n            if nums[i] + nums[j] == target:\n                return True\n    return False\n",
        domain_hint="auto",
    )

    result = service.analyze(request)

    assert result.detected_domain == "dsa"
    assert result.static_analysis.time_complexity in {"O(n^2)", "O(n^3)"}


def test_api_analyze_endpoint_returns_valid_payload() -> None:
    client = TestClient(app)
    response = client.post(
        "/analyze",
        json={
            "code": "import torch\n\ndef predict(model, x):\n    return model(x)\n",
            "context_window": "Inference helper for a classifier",
            "traceback_text": "",
            "domain_hint": "auto",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "detected_domain" in payload
    assert "score_breakdown" in payload
