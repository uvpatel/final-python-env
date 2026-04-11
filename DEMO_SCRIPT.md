# TorchReview Copilot Demo Script

## 60-90 Second Walkthrough

1. Open the Hugging Face Space and introduce TorchReview Copilot as an AI-powered Python triage assistant built with PyTorch.
2. Point to the single-sentence problem statement: teams lose time figuring out whether a failure is syntax, logic, or performance related.
3. Select the `Fix the invoice total syntax regression` example to show the app loading a real broken code sample.
4. Highlight the **Live Triage Radar** updating immediately, then call out the predicted issue class and repair risk.
5. Explain that the PyTorch layer uses CodeBERTa embeddings to compare the input against known bug patterns from the OpenEnv task catalog.
6. Scroll to the repair plan and note that the output is not just a label; it gives a prioritized remediation checklist and the nearest known failure pattern.
7. Switch to the performance example to show the confidence profile change and emphasize that the system can distinguish runtime bottlenecks from correctness bugs.
8. Close by noting that OpenEnv still powers deterministic validation under the hood, so the demo stays grounded in measurable task outcomes.
