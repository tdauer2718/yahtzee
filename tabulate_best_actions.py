from collections import Counter, defaultdict
import math
import pandas as pd
from typing import Tuple

from basic_utils import (
    ALL_ROLL_TUPLES,
    Box,
    BoxCategories,
    RollValues,
    RollAction,
    remove_dice,
    TUPLES_BY_CATEGORY,
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


def box_probability_from_action(
    roll_values: RollValues, roll_action: RollAction, box: Box
) -> float:
    if box not in BoxCategories.LowerBox:
        raise ValueError("The box must be in the lower section.")
    probability = 0
    for t in TUPLES_BY_CATEGORY[box]:
        probability += hit_probability_from_action(
            roll_values, roll_action, RollValues(*t)
        )
    return probability


def best_action_tuples_for_box(
    roll_values: RollValues, box: Box
) -> Tuple[Tuple[int, ...], float]:
    action_probabilities = {
        roll_action.dice_values_to_roll: box_probability_from_action(
            roll_values, roll_action, box
        )
        for roll_action in roll_values.roll_actions
    }
    values = max(action_probabilities, key=lambda a: action_probabilities[a])
    probability = action_probabilities[values]
    return values, probability


def all_best_action_values_for_box(box: Box) -> pd.DataFrame:
    tabulated_best_roll_actions = defaultdict(list)
    for roll_tuple in list(ALL_ROLL_TUPLES):
        dice_values_to_roll, hit_probability = best_action_tuples_for_box(
            RollValues(*roll_tuple), box
        )
        tabulated_best_roll_actions["roll_values"].append(roll_tuple)
        tabulated_best_roll_actions["dice_values_to_roll"].append(dice_values_to_roll)
        tabulated_best_roll_actions["hit_probability"].append(hit_probability)
    df = pd.DataFrame(tabulated_best_roll_actions)
    df.to_csv(f"best_roll_actions_for_{box.value}.csv")
    return df


if __name__ == "__main__":
    df_small_straight = all_best_action_values_for_box(Box.SmallStraight)
    df_large_straight = all_best_action_values_for_box(Box.LargeStraight)
