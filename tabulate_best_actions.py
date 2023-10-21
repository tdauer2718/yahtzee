from collections import Counter
import functools
import math
import pandas as pd
from typing import List, Tuple

from basic_utils import (
    ALL_ROLL_TUPLES,
    Box,
    RollValues,
    RollAction,
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


def create_expected_scores_table() -> pd.DataFrame:
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


@functools.lru_cache
def all_expected_scores_table() -> pd.DataFrame:
    df = pd.read_csv("expected_scores_by_box.csv")
    df["roll_values"] = df["roll_values"].apply(eval)
    df["dice_values_to_roll"] = df["dice_values_to_roll"].apply(eval)
    return df


@functools.lru_cache
def expected_scores_table() -> pd.DataFrame:
    df_all = pd.read_csv("expected_scores_by_box.csv")
    df_all["roll_values"] = df_all["roll_values"].apply(eval)
    df_all["dice_values_to_roll"] = df_all["dice_values_to_roll"].apply(eval)
    df = df_all.loc[
        df_all.groupby(by=["roll_values", "box"])["expected_score"].idxmax()
    ]
    df.set_index(["roll_values", "box"], inplace=True)
    return df
