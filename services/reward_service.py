"""Reward shaping logic for RL-ready code analysis scores."""

from __future__ import annotations

from schemas.response import ScoreBreakdown


class RewardService:
    """Compute reward scores from model, domain, lint, and complexity signals."""

    def compute(self, *, ml_score: float, domain_score: float, lint_score: float, complexity_penalty: float) -> ScoreBreakdown:
        """Apply dynamic reward shaping based on quality, errors, and completion."""

        quality_signal = max(0.0, min(1.0, (0.45 * ml_score) + (0.3 * domain_score) + (0.25 * lint_score)))
        error_reduction_signal = max(0.0, min(1.0, lint_score - (0.6 * complexity_penalty)))
        completion_signal = max(0.0, min(1.0, (ml_score + domain_score + lint_score) / 3.0))
        reward = max(
            0.0,
            min(
                1.0,
                (0.35 * quality_signal)
                + (0.25 * completion_signal)
                + (0.2 * error_reduction_signal)
                + (0.1 * ml_score)
                + (0.1 * domain_score)
                - (0.15 * complexity_penalty),
            ),
        )
        return ScoreBreakdown(
            ml_score=round(ml_score, 4),
            domain_score=round(domain_score, 4),
            lint_score=round(lint_score, 4),
            complexity_penalty=round(complexity_penalty, 4),
            quality_signal=round(quality_signal, 4),
            error_reduction_signal=round(error_reduction_signal, 4),
            completion_signal=round(completion_signal, 4),
            reward=round(reward, 4),
        )
