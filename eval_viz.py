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
    def __init__(self, identity, fn, num_candidates):
        self.identity = identity
        self.variations = None
        self.matching_moves = None
        self.total_time = None
        self.fn = fn
        self.moves = []
        self.num_candidates = num_candidates
        self.best_line = []

    def parse_log(self):
        with open(self.fn, 'r') as f:
            lines = f.read().splitlines()
            # Moves start on line 6
            self.variations = lines[5:-4]
            self.matching_moves = lines[-3]
            count = 1
            move_data = []
            for l in lines:
                move_data.append(l)
                if l.startswith(str(count)):
                    move_data = []
                    count += 1
                    # self.moves.append(Move(count - 2, [], []))
                elif l == "" and count > 1:
                    moves = []
                    evals = []
                    for i in range(self.num_candidates):
                        move_line = move_data[-3 - i]
                        move_line_tabs = move_line.split("\t")
                        try:
                            parsed = move_line_tabs[5].split(" ")
                            if parsed[0][:-1].isdigit():
                                move = parsed[2]
                            else:
                                move = parsed[0].split(".")[1]
                        except IndexError:
                            # Move 12
                            #print(count)
                            #print(move_data)
                            move = None
                            # There are less than 3 moves
                        if move:
                            moves.append(move)
                            evals.append(move_line_tabs[4])
                    self.moves.append(Move(count, moves, evals))
                    move_data = []

    def find_best(self):
        for count, move in enumerate(self.moves[1:]):
            best_eval = float('inf')
            if count % 2 == 0:
                best_eval = -float('inf')
            best_move = ""
            for m, e in zip(move.candidates, move.evaluations):
                evaluation = float(e[1:])
                # print(evaluation)
                sign = e[0]
                if sign == "-":
                    evaluation *= -1
                if count % 2:
                    evaluation *= - 1
                if count % 2 == 0:
                    if evaluation > best_eval:
                        best_eval = evaluation
                        best_move = m
                else:
                    if evaluation < best_eval:
                        best_eval = evaluation
                        best_move = m
            print("Move number", count)
            print("Best eval", best_eval)
            print("Best move", best_move, "\n")
            self.best_line.append(Move(count, best_move, best_eval))

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
    """Finish creating eval graph for game """
    game_tuple = namedtuple("game_tuple", ["game_id", "game_fn", "num_cand"])
    games = {"game1": game_tuple(1, "analysis/wcc18g1.log", 3)}
    wcc18g1 = Game(games["game1"].game_id, games["game1"].game_fn, games["game1"].num_cand)
    wcc18g1.parse_log()
    wcc18g1.find_best()
    # wcc18g1.create_graph()


if __name__ == "__main__":
    main()
