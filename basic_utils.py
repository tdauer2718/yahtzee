from collections import Counter
from dataclasses import dataclass
from enum import Enum
import itertools
import pandas as pd
import random
from typing import Dict, List, Optional, Set, Tuple, Union


class Box(Enum):
    Ones = "ones"
    Twos = "twos"
    Threes = "threes"
    Fours = "fours"
    Fives = "fives"
    Sixes = "sixes"
    ThreeOfAKind = "three_of_a_kind"
    FourOfAKind = "four_of_a_kind"
    FullHouse = "full_house"
    SmallStraight = "small_straight"
    LargeStraight = "large_straight"
    Yahtzee = "yahtzee"
    Chance = "chance"


@dataclass(frozen=True)
class BoxCategories:
    UpperBox = {Box.Ones, Box.Twos, Box.Threes, Box.Fours, Box.Fives, Box.Sixes}
    LowerBox = {
        Box.ThreeOfAKind,
        Box.FourOfAKind,
        Box.FullHouse,
        Box.SmallStraight,
        Box.LargeStraight,
        Box.Yahtzee,
        Box.Chance,
    }

UPPER_BOX_VALUES: Dict[Box, int] = {
    Box.Ones: 1,
    Box.Twos: 2,
    Box.Threes: 3,
    Box.Fours: 4,
    Box.Fives: 5,
    Box.Sixes: 6,
}


class RollAction:
    def __init__(self, *dice_values_to_roll: int):
        if len(dice_values_to_roll) < 1:
            raise ValueError("You must roll at least one die.")
        if len(dice_values_to_roll) > 5:
            raise ValueError("You can roll at most five dice.")
        self.dice_values_to_roll = dice_values_to_roll

    def __repr__(self):
        return f"RollAction(dice_values_to_roll={self.dice_values_to_roll})"


@dataclass
class ScoreAction:
    score: int
    dice_values: Tuple[int, ...]
    box: Box


class RollValues:
    def __init__(self, *values: int):
        if len(values) != 5:
            raise ValueError("There must be exactly five dice present.")
        if not set(values).issubset(set(range(1, 7))):
            raise ValueError("Die values must be integers from 1 to 6.")
        self.values = tuple(sorted(values))
        self.value_counts = dict(Counter(self.values))

    def __repr__(self):
        return f"Dice values = {', '.join([str(v) for v in self.values])}"

    def checks_box(self, box: Box) -> bool:
        if box == Box.ThreeOfAKind:
            return any([v >= 3 for v in self.value_counts.values()])
        elif box == Box.FourOfAKind:
            return any([v >= 4 for v in self.value_counts.values()])
        elif box == Box.FullHouse:
            return set(self.value_counts.values()) == {2, 3}
        elif box == Box.SmallStraight:
            values = set(self.values)
            return (
                {1, 2, 3, 4}.issubset(values)
                or {2, 3, 4, 5}.issubset(values)
                or {3, 4, 5, 6}.issubset(values)
            )
        elif box == Box.LargeStraight:
            return tuple(sorted(set(self.values))) in {(1, 2, 3, 4, 5), (2, 3, 4, 5, 6)}
        elif box == Box.Yahtzee:
            return len(set(self.values)) == 1
        elif box in [Box.Chance] + list(BoxCategories.UpperBox):
            return True
        else:
            raise ValueError(
                "Box must be one of the 13 boxes on the scorecard."
            )

    def mode_value(self, tiebreak_by_highest_value: bool = True) -> int:
        if tiebreak_by_highest_value:
            mode = max(self.values, key=lambda k: (self.value_counts[k], k))
        else:
            mode = max(self.values, key=lambda k: self.value_counts[k])
        return mode

    def score_from_box(self, box: Box) -> int:
        """
        Gives the score for this roll when `box` is chosen.
        """
        for upper_box in BoxCategories.UpperBox:
            if box == upper_box:
                value = UPPER_BOX_VALUES[box]
                return self.value_counts[value] * value if value in self.value_counts else 0
        if box == Box.ThreeOfAKind:
            if self.checks_box(Box.ThreeOfAKind):
                v = [v for v, c in self.value_counts.items() if c >= 3][0]
                score = self.value_counts[v] * v
            else:
                score = 0
            return score
        if box == Box.FourOfAKind:
            if self.checks_box(Box.FourOfAKind):
                v = [v for v, c in self.value_counts.items() if c >= 4][0]
                score = self.value_counts[v] * v
            else:
                score = 0
            return score
        if box == Box.FullHouse:
            return 25 if self.checks_box(Box.FullHouse) else 0
        if box == Box.SmallStraight:
            return 30 if self.checks_box(Box.SmallStraight) else 0
        if box == Box.LargeStraight:
            return 40 if self.checks_box(Box.LargeStraight) else 0
        if box == Box.Yahtzee:
            return 50 if self.checks_box(Box.Yahtzee) else 0
        if box == Box.Chance:
            return sum(self.values)
        else:
            raise ValueError("Please choose one of the 13 boxes on the scorecard.")

    @property
    def score_actions(self) -> List[ScoreAction]:
        """
        Gives the possible score actions for a roll, acting as if the ScoreCard is empty.
        """
        results = []

        for box in BoxCategories.UpperBox:
            score = self.score_from_box(box)
            value = UPPER_BOX_VALUES[box]
            dice_values = (
                self.value_counts[value] * (value,)
                if value in self.value_counts
                else ()
            )
            results.append(ScoreAction(score, dice_values, box))

        if self.checks_box(Box.ThreeOfAKind):
            score = self.score_from_box(Box.ThreeOfAKind)
            v = [v for v, c in self.value_counts.items() if c >= 3][0]
            dice_values = self.value_counts[v] * (v,)
            box = Box.ThreeOfAKind
            results.append(ScoreAction(score, dice_values, box))
        else:
            results.append(ScoreAction(0, (), Box.ThreeOfAKind))

        if self.checks_box(Box.FourOfAKind):
            score = self.score_from_box(Box.FourOfAKind)
            v = [v for v, c in self.value_counts.items() if c >= 4][0]
            dice_values = self.value_counts[v]  * (v,)
            box = Box.FourOfAKind
            results.append(ScoreAction(score, dice_values, box))
        else:
            results.append(ScoreAction(0, (), Box.FourOfAKind))

        if self.checks_box(Box.FullHouse):
            score = self.score_from_box(Box.FullHouse)
            dice_values = tuple(
                v for v in self.values
            )  # copy instead of reference (perhaps unnecessary)
            box = Box.FullHouse
            results.append(ScoreAction(score, dice_values, box))
        else:
            results.append(ScoreAction(0, (), Box.FullHouse))

        if self.checks_box(Box.SmallStraight):
            score = self.score_from_box(Box.SmallStraight)
            box = Box.SmallStraight
            if self.checks_box(Box.LargeStraight):
                dice_values = tuple(v for v in self.values)
            else:
                dice_values = tuple(
                    [
                        s
                        for s in [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]
                        if s.issubset(self.values)
                    ][0]
                )
            results.append(ScoreAction(score, dice_values, box))
        else:
            results.append(ScoreAction(0, (), Box.SmallStraight))

        if self.checks_box(Box.LargeStraight):
            score = self.score_from_box(Box.LargeStraight)
            dice_values = tuple(v for v in self.values)
            box = Box.LargeStraight
            results.append(ScoreAction(score, dice_values, box))
        else:
            results.append(ScoreAction(0, (), Box.LargeStraight))

        if self.checks_box(Box.Yahtzee):
            # For simplicity, ignore Yahtzee bonus and joker rules for now.
            # Later I will likely add that functionality at the GameState
            # level instead of here, since I'd prefer that the RollValues class
            # not depend on ScoreCard. (I may change my mind on this.)
            score = self.score_from_box(Box.Yahtzee)
            v = self.values[0]
            dice_values = 5 * (v,)
            box = Box.Yahtzee
            results.append(ScoreAction(score, dice_values, box))
        else:
            results.append(ScoreAction(0, (), Box.Yahtzee))

        # Chance
        score = self.score_from_box(Box.Chance)
        dice_values = tuple(v for v in self.values)
        box = Box.Chance
        results.append(ScoreAction(score, dice_values, box))

        return results

    @property
    def roll_actions(self) -> List[RollAction]:
        roll_action_tuples = {
            values
            for values in itertools.chain.from_iterable(
                [list(itertools.combinations(self.values, i)) for i in range(1, 6)]
            )
        }
        return [RollAction(*values) for values in roll_action_tuples]


ALL_ROLL_TUPLES = {t for t in itertools.combinations_with_replacement(range(1, 7), 5)}
ROLL_TUPLES_BY_BOX = {
    box: {t for t in ALL_ROLL_TUPLES if RollValues(*t).checks_box(box)}
    for box in Box
}


class ScoreCard:
    def __init__(self):
        # For each string in the Box enum, make an attribute in this class
        self.ones: Optional[int] = None
        self.twos: Optional[int] = None
        self.threes: Optional[int] = None
        self.fours: Optional[int] = None
        self.fives: Optional[int] = None
        self.sixes: Optional[int] = None
        self.three_of_a_kind: Optional[int] = None
        self.four_of_a_kind: Optional[int] = None
        self.full_house: Optional[int] = None
        self.small_straight: Optional[int] = None
        self.large_straight: Optional[int] = None
        self.yahtzee: Optional[int] = None
        self.chance: Optional[int] = None
        # Maybe this should be in a test, but I'll put it here
        # for now (makes sure Box and the attributes above match)
        if set(vars(self).keys()) != {b.value for b in Box}:
            raise ValueError("Attributes of ScoreCard must match Box enum values.")

    def __repr__(self):
        return pd.DataFrame(
            self.all_scores_dict.items(), columns=["Category", "Value"]
        ).to_string(index=False)

    @property
    def top_scores_dict(self) -> Dict[str, Optional[int]]:
        return {
            "ones": self.ones,
            "twos": self.twos,
            "threes": self.threes,
            "fours": self.fours,
            "fives": self.fives,
            "sixes": self.sixes,
        }

    @property
    def bottom_scores_dict(self) -> Dict[str, Optional[int]]:
        return {
            "three_of_a_kind": self.three_of_a_kind,
            "four_of_a_kind": self.four_of_a_kind,
            "full_house": self.full_house,
            "small_straight": self.small_straight,
            "large_straight": self.large_straight,
            "yahtzee": self.yahtzee,
            "chance": self.chance,
        }

    @property
    def all_scores_dict(self) -> Dict[str, Optional[int]]:
        return {**self.top_scores_dict, **self.bottom_scores_dict}

    @property
    def top_score(self) -> int:
        partial_sum = sum(
            [s if s is not None else 0 for s in self.top_scores_dict.values()]
        )
        return partial_sum + 35 if partial_sum >= 63 else partial_sum

    @property
    def bottom_score(self) -> int:
        return sum(
            [s if s is not None else 0 for s in self.bottom_scores_dict.values()]
        )

    @property
    def score(self) -> int:
        return self.top_score + self.bottom_score

    @property
    def game_finished(self) -> bool:
        return all(self.all_scores_dict.values())

    @property
    def unused_boxes(self) -> Set[Box]:
        return {b for b in Box if getattr(self, b.value) is None}

    def fill_in_box(self, box: Box, score: int):
        setattr(self, box.value, score)


def roll_first() -> RollValues:
    return RollValues(*random.choices(population=range(1, 7), k=5))


def remove_dice(
    roll_values: RollValues, values_to_remove: Tuple[int, ...]
) -> List[int]:
    return list((Counter(roll_values.values) - Counter(values_to_remove)).elements())


def roll_again(roll_values: RollValues, roll_action: RollAction) -> RollValues:
    if len(Counter(roll_action.dice_values_to_roll) - Counter(roll_values.values)) != 0:
        # It's not enough to do a subset check; consider the case where (1, 2, 3, 4, 5)
        # are the current values and (1, 1) are the (erroneous) values to re-roll.
        raise ValueError("You cannot re-roll dice that are not present.")
    values_that_stay = remove_dice(roll_values, roll_action.dice_values_to_roll)
    new_values = random.choices(
        population=range(1, 7), k=len(roll_action.dice_values_to_roll)
    )
    return RollValues(*(values_that_stay + new_values))


class GameState:
    def __init__(self, narrate: bool = True):
        self.narrate = narrate
        self.scorecard = ScoreCard()
        self.turn_started = False

    def start_turn(self):
        self.roll_values = roll_first()
        self.rolls_completed = 1
        if self.narrate:
            print(f"Roll number {self.rolls_completed} gives:")
            print(self.roll_values)
        self.turn_started = True

    @property
    def possible_score_actions(self) -> List[ScoreAction]:
        """
        Uses scorecard and roll_values to return the possible ScoreAction values.
        """
        return sorted(
            [
                a
                for a in self.roll_values.score_actions
                if a.box in self.scorecard.unused_boxes
            ],
            key=lambda x: x.score,
            reverse=True,
        )

    @property
    def possible_roll_actions(self) -> List[RollAction]:
        """
        Uses scorecard and roll_values to return the possible RollAction values.
        Given how easy it is to determine RollAction values from a list of dice,
        this should be implemented in a way that requires less re-computation
        when we call possible_actions()... but for now I'll avoid premature optimization.
        """
        return self.roll_values.roll_actions if self.rolls_completed < 3 else []

    @property
    def possible_actions(self) -> List[Union[RollAction, ScoreAction]]:
        """
        Uses scorecard, roll_values, and rolls_completed to determine the possible actions,
        which can be RollAction or ScoreAction values.
        """
        return self.possible_score_actions + self.possible_roll_actions

    def re_roll(self, *dice_values_to_roll: int):
        """
        Rolls the dice specified (by value) and updates roll_values and rolls_completed.
        """
        if not self.turn_started:
            raise ValueError(
                "Please start your turn using the `start_turn()` method first."
            )
        roll_action = RollAction(*dice_values_to_roll)
        if self.rolls_completed >= 3:
            raise ValueError(
                "Your three rolls for this turn have already been completed."
            )
        self.roll_values = roll_again(self.roll_values, roll_action)
        self.rolls_completed += 1
        if self.narrate:
            print(
                f"Roll number {self.rolls_completed} (re-rolling {', '.join([str(v) for v in dice_values_to_roll])}) gives:"
            )
            print(self.roll_values)

    def update_score(self, box: Box):
        if not self.turn_started:
            raise ValueError(
                "Please start your turn using the `start_turn()` method first."
            )
        # will need to update when I implement joker rules
        try:
            score = [a for a in self.possible_score_actions if a.box == box][0].score
        except IndexError:
            raise ValueError("You are trying to use a box that is already filled in.")
        self.scorecard.fill_in_box(box, score)
        self.turn_started = False
        if self.narrate:
            print(f"Filling in {box.name}, with value {score}\n")

    @property
    def best_scores(self, n: int = 3, print_only=True):
        if print_only:
            for a in self.possible_score_actions[:n]:
                print(a)
        else:
            return self.possible_score_actions[:n]

    def take_action(self, action: Union[ScoreAction, RollAction]):
        if isinstance(action, ScoreAction):
            self.update_score(action.box)
        elif isinstance(action, RollAction):
            self.re_roll(*action.dice_values_to_roll)
        else:
            raise ValueError(
                "action must be an instance of `ScoreAction` or `RollAction`."
            )
