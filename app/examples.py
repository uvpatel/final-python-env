"""Example snippets for each supported analysis domain."""

from __future__ import annotations


EXAMPLES = {
    "DSA": {
        "domain_hint": "dsa",
        "context_window": "Competitive-programming helper for pair lookup on large arrays.",
        "traceback_text": "",
        "code": """def two_sum(nums, target):\n    for i in range(len(nums)):\n        for j in range(i + 1, len(nums)):\n            if nums[i] + nums[j] == target:\n                return [i, j]\n    return []\n""",
    },
    "Data Science": {
        "domain_hint": "data_science",
        "context_window": "Feature engineering step in a churn-prediction notebook.",
        "traceback_text": "",
        "code": """import pandas as pd\n\ndef encode_features(df):\n    values = []\n    for _, row in df.iterrows():\n        values.append(row['age'] * row['sessions'])\n    df['score'] = values\n    return df\n""",
    },
    "ML / DL": {
        "domain_hint": "ml_dl",
        "context_window": "Inference utility for a PyTorch classifier used in a batch review job.",
        "traceback_text": "",
        "code": """import torch\n\nclass Predictor:\n    def __init__(self, model):\n        self.model = model\n\n    def predict(self, batch):\n        outputs = self.model(batch)\n        return outputs.argmax(dim=1)\n""",
    },
    "Web / FastAPI": {
        "domain_hint": "web",
        "context_window": "Backend endpoint for creating review tasks from user-submitted payloads.",
        "traceback_text": "",
        "code": """from fastapi import FastAPI, Request\n\napp = FastAPI()\n\n@app.post('/tasks')\ndef create_task(request: Request):\n    payload = request.json()\n    return {'task': payload}\n""",
    },
}
