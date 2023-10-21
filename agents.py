from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import List, Union

from basic_utils import GameState, RollAction, ScoreAction, ScoreCard


class Agent(ABC):
    def __init__(self, narrate: bool = False):
        self.narrate = narrate

    def play_single_game(self) -> ScoreCard:
        game_state = GameState(narrate=self.narrate)
        for _ in range(13):
            game_state.start_turn()
            while game_state.turn_started:
                action = self.choose_action(game_state)
                game_state.take_action(action)
        if self.narrate:
            print("Final scorecard:\n")
            print(game_state.scorecard, end="\n\n")
            print(f"Final score = {game_state.scorecard.score}")
        return game_state.scorecard

    @abstractmethod
    def choose_action(self, game_state: GameState) -> Union[RollAction, ScoreAction]:
        """Use self.game_state to choose a possible action."""
        ...

    def play_games(
        self, n_games: int, plot_results: bool = True, histogram_bins: int = 20
    ) -> List[int]:
        """
        Plays the specified number of games with the Agent, returning the Agent's final score
        for each game and optionally plotting the results in a histogram.
        """
        agent_scores = []
        for _ in range(n_games):
            scorecard = self.play_single_game()
            agent_scores.append(scorecard.score)
        if plot_results:
            plt.hist(agent_scores, bins=histogram_bins)
            plt.xlabel("Score")
            plt.ylabel("Number of occurrences")
            plt.title(f"Scores from {n_games} Yahtzee games with {type(self).__name__}")
            plt.show()
            print(f"Mean score = {np.mean(agent_scores):.3g}")
            print(f"Standard deviation = {np.std(agent_scores):.3g}")
            print(f"Median score = {np.median(agent_scores):.3g}")

        return agent_scores


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
    reward. That is, on each turn it always picks the highest score
    after just the first roll.
    """

    def choose_action(self, game_state: GameState) -> ScoreAction:
        return game_state.sorted_possible_score_actions[0]


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
            action = game_state.sorted_possible_score_actions[0]
        return action
