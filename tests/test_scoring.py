from __future__ import annotations

from graders import grade_task
from graders.shared import component_score, final_score_pipeline, safe_score, shaped_score
from models import PythonCodeReviewAction
from server.env import PythonCodeReviewEnvironment
from tasks import list_tasks


def assert_open_unit_interval(value: float) -> None:
    assert 0 < value < 1, f"Invalid score: {value}"


def test_score_helpers_clamp_extremes_into_open_interval() -> None:
    for value in (0.0, 1.0, -999999.0, 999999.0):
        assert_open_unit_interval(safe_score(value))
        assert_open_unit_interval(final_score_pipeline(value))

    for progress in (0.0, 0.5, 1.0):
        assert_open_unit_interval(shaped_score(progress))
        assert_open_unit_interval(component_score(progress))


def test_task_grades_stay_strictly_between_zero_and_one() -> None:
    for task in list_tasks():
        starter_grade = grade_task(task, task.starter_code, include_hidden=False)
        reference_grade = grade_task(task, task.reference_code, include_hidden=True)

        for grade in (starter_grade, reference_grade):
            assert_open_unit_interval(grade.score)
            assert_open_unit_interval(grade.syntax_score)
            assert_open_unit_interval(grade.quality_score)
            assert_open_unit_interval(grade.runtime_score)


def test_environment_scores_and_rewards_stay_in_open_interval() -> None:
    env = PythonCodeReviewEnvironment(verbose=False)
    observation = env.reset(task_id="bug_fix_session_windows")

    assert_open_unit_interval(observation.score)
    assert_open_unit_interval(observation.reward_details.value)

    no_op_action = PythonCodeReviewAction(action_type="edit_code", code=observation.current_code)
    next_observation, reward, _, _ = env.step_result(no_op_action)
    assert_open_unit_interval(next_observation.score)
    assert_open_unit_interval(reward)
    assert_open_unit_interval(next_observation.reward_details.value)

    submit_action = PythonCodeReviewAction(action_type="submit_solution", code=env._task.reference_code)
    final_observation, final_reward, _, _ = env.step_result(submit_action)
    assert_open_unit_interval(final_observation.score)
    assert_open_unit_interval(final_reward)
    assert_open_unit_interval(final_observation.reward_details.value)
