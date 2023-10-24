from collections import Counter
import functools
import math
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple, Union

from basic_utils import (
    ALL_ROLL_TUPLES,
    Box,
    BoxCategories,
    RollValues,
    RollAction,
    ScoreAction,
    remove_dice,
    ROLL_TUPLES_BY_BOX,
)


def probability_of_rolling_values(*values: int):
    value_counts = Counter(values)
    probability = math.factorial(len(values)) * (1 / 6) ** len(values)
    for _, count in value_counts.items():
        probability /= math.factorial(count)
    return probability


def hit_probability_from_action(
    roll_values: RollValues,
    roll_action: RollAction,
    desired_next_roll_values: RollValues,
) -> float:
    values_that_stay = tuple(remove_dice(roll_values, roll_action.dice_values_to_roll))
    values_needed_on_roll = remove_dice(desired_next_roll_values, values_that_stay)
    if len(values_needed_on_roll) == len(roll_action.dice_values_to_roll):
        return probability_of_rolling_values(*values_needed_on_roll)
    else:
        # In this case we can't achieve the desired values using roll_action
        return 0


def create_expected_scores_table_one_roll() -> pd.DataFrame:
    roll_tuples: List[Tuple[int, ...]] = []
    boxes: List[str] = []
    roll_action_tuples: List[Tuple[int, ...]] = []
    expected_scores: List[float] = []

    for roll_tuple in ALL_ROLL_TUPLES:
        roll_values = RollValues(*roll_tuple)
        roll_actions = roll_values.roll_actions
        for box in Box:
            for roll_action in roll_actions:
                # Probabilities of getting the desired roll values:
                hit_probabilities = [
                    hit_probability_from_action(
                        roll_values, roll_action, RollValues(*desired_roll_tuple)
                    )
                    for desired_roll_tuple in ROLL_TUPLES_BY_BOX[box]
                ]
                # Scores of the desired rolls:
                scores = [
                    RollValues(*desired_roll_tuple).score_from_box(box)
                    for desired_roll_tuple in ROLL_TUPLES_BY_BOX[box]
                ]
                expected_score = sum([p * s for p, s in zip(hit_probabilities, scores)])
                roll_tuples.append(roll_tuple)
                roll_action_tuples.append(roll_action.dice_values_to_roll)
                boxes.append(box.name)
                expected_scores.append(expected_score)
            # Include the case where no dice are rolled
            roll_tuples.append(roll_tuple)
            boxes.append(box.name)
            roll_action_tuples.append(tuple())
            expected_scores.append(roll_values.score_from_box(box))

    results = {
        "roll_values": roll_tuples,
        "box": boxes,
        "dice_values_to_roll": roll_action_tuples,
        "expected_score": expected_scores,
    }
    df = pd.DataFrame(results)
    df.to_csv("expected_scores_by_box.csv", index=False)
    return df


def create_expected_scores_table_two_rolls():
    roll_tuples: List[Tuple[int, ...]] = []
    boxes: List[str] = []
    roll_action_tuples: List[Tuple[int, ...]] = []
    expected_scores: List[float] = []

    # Start with roll_values. For each possible roll_action on these values,
    # find the resulting `roll_tuple_result_from_roll_action`s and their
    # probabilities. For each of these, find the expected score from the
    # best roll_action toward the desired box. Then add up the probabilities
    # times these values to find the expected score in the desired box
    # for each of the original roll_values.

    for roll_tuple in ALL_ROLL_TUPLES:
        roll_values = RollValues(*roll_tuple)
        roll_actions = roll_values.roll_actions
        for box in Box:
            for roll_action in roll_actions:
                roll_tuples_after_roll_action = []
                ps = (
                    []
                )  # probability of getting each roll_tuple_result_from_roll_action
                for roll_tuple_result_from_roll_action in ALL_ROLL_TUPLES:
                    p = hit_probability_from_action(
                        roll_values,
                        roll_action,
                        RollValues(*roll_tuple_result_from_roll_action),
                    )
                    if p == 0:
                        continue
                    roll_tuples_after_roll_action.append(
                        roll_tuple_result_from_roll_action
                    )
                    ps.append(p)
                # Expected score for the best roll_action for each roll_tuple_result_from_roll_action:
                expected_scores_of_roll_tuples_after_best_roll_action = [
                    best_expected_scores_table_by_box(n_rolls_left=1)
                    .loc[t, :]
                    .loc[box.name]
                    .expected_score
                    for t in roll_tuples_after_roll_action
                ]
                score = sum(
                    [
                        p * s
                        for p, s in zip(
                            ps, expected_scores_of_roll_tuples_after_best_roll_action
                        )
                    ]
                )
                roll_tuples.append(roll_tuple)
                boxes.append(box.name)
                roll_action_tuples.append(roll_action.dice_values_to_roll)
                expected_scores.append(score)
            # Include the case where no dice are rolled
            roll_tuples.append(roll_tuple)
            boxes.append(box.name)
            roll_action_tuples.append(tuple())
            expected_scores.append(roll_values.score_from_box(box))

    results = {
        "roll_values": roll_tuples,
        "box": boxes,
        "dice_values_to_roll": roll_action_tuples,
        "expected_score": expected_scores,
    }
    df = pd.DataFrame(results)
    df.to_csv("expected_scores_by_box_two_rolls.csv", index=False)
    return df


@functools.lru_cache
def all_expected_scores_table_by_box(n_rolls_left: int) -> pd.DataFrame:
    if n_rolls_left == 1:
        df = pd.read_csv("expected_scores_by_box_one_roll.csv")
    elif n_rolls_left == 2:
        df = pd.read_csv("expected_scores_by_box_two_rolls.csv")
    else:
        raise ValueError("n_rolls_left must be 1 or 2.")
    df["roll_values"] = df["roll_values"].apply(eval)
    df["dice_values_to_roll"] = df["dice_values_to_roll"].apply(eval)
    return df


@functools.lru_cache
def best_expected_scores_table_by_box(n_rolls_left: int) -> pd.DataFrame:
    df_all = all_expected_scores_table_by_box(n_rolls_left)
    df = df_all.loc[
        df_all.groupby(by=["roll_values", "box"])["expected_score"].idxmax()
    ]
    df.set_index(["roll_values", "box"], inplace=True)
    return df


def best_roll_action_for_box_with_score(
    roll_values: RollValues,
    box: Box,
    n_rolls_left: int,
    upper_box_multiplier: float = 1.0,
) -> Tuple[Optional[RollAction], float]:
    """
    Given a set of dice values and a box, returns the dice to roll in order to maximize
    the expected value of the score in the box, and also gives the expecte score. If
    rolling no dice gives the best score, then None is returned as the roll action.
    n_rolls_left = 1 gives the result if only one roll is allowed, and n_rolls_left = 2 allows
    for 2 rolls.
    """
    row = (
        best_expected_scores_table_by_box(n_rolls_left=n_rolls_left)
        .loc[roll_values.values, :]
        .loc[box.name]
    )
    roll_action_tuple, score = row["dice_values_to_roll"], row["expected_score"]
    roll_action = (
        RollAction(*roll_action_tuple) if len(roll_action_tuple) != 0 else None
    )
    return (
        roll_action,
        upper_box_multiplier * score if box in BoxCategories.UpperBox else score,
    )


def best_roll_action_for_box(
    roll_values: RollValues, box: Box, n_rolls_left: int, upper_box_multiplier: float = 1.0
) -> Optional[RollAction]:
    """
    Given a set of dice values and a box, returns the dice to roll in order to maximize
    the expected value of the score in the box. If rolling no dice gives the best score,
    then None is returned.
    """
    return best_roll_action_for_box_with_score(
        roll_values,
        box,
        upper_box_multiplier=upper_box_multiplier,
        n_rolls_left=n_rolls_left,
    )[0]


def best_action_by_box_with_score(
    roll_values: RollValues,
    allowed_boxes: Set[Box],
    n_rolls_left: int,
    upper_box_multiplier: float = 1.0,
) -> Dict[Box, Tuple[Union[RollAction, ScoreAction], float]]:
    result = {}
    for box in allowed_boxes:
        if n_rolls_left in {1, 2}:
            best_roll_action, score = best_roll_action_for_box_with_score(
                roll_values, box, n_rolls_left=n_rolls_left, upper_box_multiplier=upper_box_multiplier
            )
            best_action = (
                best_roll_action
                if best_roll_action is not None
                else ScoreAction(roll_values.score_from_box(box), roll_values.values, box)
            )
        elif n_rolls_left == 0:
            score = roll_values.score_from_box(box)
            best_action = ScoreAction(score, roll_values.values, box)
        else:
            raise ValueError("n_rolls_left must be 0, 1 or 2.")
        result[box] = (best_action, score)
    return result


def greedy_best_action(
    roll_values: RollValues,
    allowed_boxes: Set[Box],
    n_rolls_left: int,
    upper_box_multiplier: float = 1.0,
) -> Union[RollAction, ScoreAction]:
    return max(
        best_action_by_box_with_score(
            roll_values,
            allowed_boxes,
            n_rolls_left=n_rolls_left,
            upper_box_multiplier=upper_box_multiplier,
        ).values(),
        key=lambda x: x[1],
    )[0]
