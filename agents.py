import random
from typing import Union

from basic_utils import Agent, GameState, RollAction, ScoreAction


class RandomAgent(Agent):
    """
    A completely random gameplay agent. At each timestep, it simply
    chooses one of the allowed actions, uniformly at random.
    It plays quite poorly, as often it chooses a `ScoreAction`
    with score 0 even when there are easy better options.
    """

    def choose_action(self, game_state: GameState) -> Union[RollAction, ScoreAction]:
        return random.choice(game_state.possible_actions)


class GreedyAgent(Agent):
    """
    A completely greedy gameplay agent. Given a game state,
    it always selects the action that gives the highest immediate
    reward. That is, on each turn it always picks the highest score after just the first roll.
    """

    def choose_action(self, game_state: GameState) -> ScoreAction:
        return game_state.possible_score_actions[0]


class EpsilonGreedyAgent(Agent):
    def __init__(self, epsilon: float):
        if epsilon < 0 or epsilon >= 1:
            raise ValueError("epsilon must be >= 0 and < 1.")
        self.epsilon = epsilon
        super().__init__()

    def choose_action(self, game_state: GameState) -> Union[RollAction, ScoreAction]:
        r = random.random()
        if r < self.epsilon and len(game_state.possible_roll_actions) != 0:
            action = random.choice(game_state.possible_roll_actions)
        else:
            action = game_state.possible_score_actions[0]
        return action
