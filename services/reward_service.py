"""Reward shaping logic for RL-ready code analysis scores."""

from __future__ import annotations

from schemas.response import ScoreBreakdown


class RewardService:
    """Compute reward scores from model, domain, lint, and complexity signals."""

    def compute(self, *, ml_score: float, domain_score: float, lint_score: float, complexity_penalty: float) -> ScoreBreakdown:
        """Apply the weighted reward formula and clamp the result."""

        reward = max(
            0.0,
            min(
                1.0,
                (0.4 * ml_score) + (0.2 * domain_score) + (0.2 * lint_score) - (0.2 * complexity_penalty),
            ),
        )
        return ScoreBreakdown(
            ml_score=round(ml_score, 4),
            domain_score=round(domain_score, 4),
            lint_score=round(lint_score, 4),
            complexity_penalty=round(complexity_penalty, 4),
            reward=round(reward, 4),
        )
