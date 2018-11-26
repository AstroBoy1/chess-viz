# Michael Omori
# Deep Learning
# Engine Visualization

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random
from collections import namedtuple


class Move:
    def __init__(self, number, candidates, evaluations):
        self.number = number
        self.candidates = candidates
        self.evaluations = evaluations


class Game:
    def __init__(self, identity, fn):
        self.identity = identity
        self.variations = None
        self.matching_moves = None
        self.total_time = None
        self.fn = fn
        self.moves = []

    def parse_log(self):
        with open(self.fn, 'r') as f:
            lines = f.read().splitlines()
            # Moves start on line 6
            self.variations = lines[5:-4]
            self.matching_moves = lines[-3]
            count = 1
            for l in lines:
                if l.startswith(str(count)):
                    # number = l.rstrip()
                    move = Move(count)
                    count += 1
                    self.moves.append(move)
        # print("Variations \n", self.variations)
        print("Moves", self.moves)

    def create_graph(self):
        """Creates the evaluation graph for a given game"""
        sns.set(style="white", context="talk")
        f, (ax1) = plt.subplots(1, 1, figsize=(10, 5))
        df = pd.DataFrame()
        length = 10
        moves = ["Move " + str(num) for num in range(length)]
        evals = [random.randrange(-10, 10) for _ in range(length)]
        df['moves'] = moves
        df['eval'] = evals
        sns.barplot(x="moves", y="eval", palette="rocket", ax=ax1, data=df)
        ax1.axhline(0, color="k", clip_on=False)
        ax1.set_xlabel("Move #")
        ax1.set_ylabel("Evaluation")
        ax1.set_title("LC0 Evaluation of the Chess Game")
        sns.despine(bottom=True)
        plt.setp(f.axes, yticks=list(range(min(evals), max(evals))))
        plt.tight_layout(h_pad=2)
        plt.show()
        return ax1


def main():
    """Finish creating eval graph for game 1, line 38"""
    game_tuple = namedtuple("game_tuple", ["game_id", "game_fn"])
    games = {"game1": game_tuple(1, "analysis/Analyses.log")}
    wcc18g1 = Game(games["game1"].game_id, games["game1"].game_fn)
    wcc18g1.parse_log()
    # wcc18g1.create_graph()


if __name__ == "__main__":
    main()
