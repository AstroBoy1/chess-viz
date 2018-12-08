# Michael Omori
# Deep Learning
# Engine Visualization

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random
from collections import namedtuple
import statistics
import math
from scipy.stats import lognorm
import string


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
        self.mean = None
        self.variance = None
        self.stdv = None
        self.white_loss = 0
        self.black_loss = 0
        self.white_differences = []
        self.black_differences = []

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
            """TODO: hotfix"""
            if best_eval == -float('inf') or best_eval == float('inf'):
                best_eval = 0
            #print("Move number", count)
            #print("Best eval", best_eval)
            #print("Best move", best_move, "\n")
            self.best_line.append(Move(count, best_move, best_eval))

    def calculate_loss(self):
        for i in range(0, len(self.best_line) - 1):
            current_eval = self.best_line[i].evaluations
            next_eval = self.best_line[i + 1].evaluations
            ds = abs(current_eval - next_eval)
            # print(ds)
            if i % 2:
                self.black_loss += ds
                self.black_differences.append(ds)
            else:
                self.white_loss += ds
                self.white_differences.append(ds)

    def create_graph(self):
        """Creates the evaluation graph for a given game"""
        sns.set(style="white", context="talk")
        f, (ax1) = plt.subplots(1, 1, figsize=(10, 5))
        df = pd.DataFrame()
        length = len(self.best_line)
        moves = ["Move " + str(num) for num in range(length)]
        evals = [move.evaluations for move in self.best_line]
        self.mean = statistics.mean(evals)
        self.stdv = statistics.stdev(evals)
        self.variance = statistics.variance(evals)
        #print("Mean evaluation", self.mean)
        #print("Variance", self.variance)
        #print("Standard deviation", self.stdv)
        df['moves'] = moves
        df['eval'] = evals
        sns.barplot(x="moves", y="eval", palette="rocket", ax=ax1, data=df)
        sns.despine()
        ax1.axhline(0, color="k", clip_on=False)
        ax1.set_xlabel("Move #")
        ax1.set_ylabel("Evaluation")
        ax1.set_title("LC0 Evaluation of Game" + str(self.identity))
        sns.despine(bottom=True)
        plt.setp(f.axes, yticks=list(range(-5, 5)))
        plt.setp(f.axes, xticks=list(range(0, len(self.best_line))))
        plt.tight_layout(h_pad=2)
        ax1.axes.get_xaxis().set_visible(False)
        plt.savefig("output/Eval" + str(self.identity) + ".png")
        return ax1

    def output_stats(self):
        # print("Average White loss", self.white_loss / len(self.best_line) / 2)
        # print("Average Black loss", self.black_loss / len(self.best_line) / 2)
        # print("Black differences", sorted(self.black_differences))
        # print("White differences", sorted(self.white_differences))
        # print("White loss standard deviation", statistics.stdev(self.white_differences))
        # print("Black loss standard deviation", statistics.stdev(self.black_differences))
        # print("Best line", [move.evaluations for move in self.best_line])

        plt.rcParams["figure.figsize"] = [20, 10]
        bins = [x * 0.25 for x in range(10)]
        f, (ax1, ax2) = plt.subplots(2, 1, sharex='col')
        plt.tight_layout()
        ax1.hist(self.white_differences, color='yellow', bins=bins)
        ax1.set_title('White Loss Game' + str(self.identity))
        ax2.hist(self.black_differences, color='black', bins=bins)
        ax2.set_title("Black Loss Game" + str(self.identity))
        plt.savefig("output/Losses" + str(self.identity) + ".png")


def main():
    """Bug in getting inf for best line eval"""
    carlsen_loss = []
    caruana_loss = []
    for game_number in range(1, 13):
        print("\n", "Game Number " + str(game_number))
        game_tuple = namedtuple("game_tuple", ["game_id", "game_fn", "num_cand"])
        games = {"game" + str(game_number): game_tuple(game_number, "analysis/wcc18g" + str(game_number) + ".log", 3)}
        # games = {"game" + str(game_number): game_tuple(game_number, "analysis/wcc18g1.log", 3)}
        wcc18g1 = Game(games["game" + str(game_number)].game_id, games["game" + str(game_number)].game_fn, games["game" + str(game_number)].num_cand)
        wcc18g1.parse_log()
        wcc18g1.find_best()
        wcc18g1.calculate_loss()
        wcc18g1.create_graph()
        wcc18g1.output_stats()
        # Caruana is white
        if game_number % 2:
            caruana_loss.append(wcc18g1.white_differences)
            carlsen_loss.append(wcc18g1.black_differences)
        else:
            caruana_loss.append(wcc18g1.black_differences)
            carlsen_loss.append(wcc18g1.white_differences)
    carlsen_flat_list = [item for sublist in carlsen_loss for item in sublist]
    caruana_flat_list = [item for sublist in caruana_loss for item in sublist]
    #print("Carlsen loss: ", carlsen_flat_list)
    #print("Carauana loss: ", caruana_flat_list)
    loss_df = pd.DataFrame()
    loss_df['carlsen_loss'] = carlsen_flat_list
    loss_df['caruana_loss'] = caruana_flat_list
    loss_df.to_csv("output/OverallLoses.csv")


if __name__ == "__main__":
    main()
