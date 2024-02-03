import itertools
import json
import csv
import os
import datetime
import numpy as np
import pulp as plp


class MMA_Optimizer:
    site = None
    config = None
    problem = None
    output_dir = None
    num_lineups = None
    num_uniques = None
    lineups = []
    player_dict = {}
    at_least = {}
    at_most = {}
    exactly = {}
    projection_minimum = 0
    randomness_amount = 0
    fighter_list = []
    allow_fighters_from_same_fight = False

    def __init__(self, site=None, num_lineups=0, num_uniques=1):
        self.site = site
        self.num_lineups = int(num_lineups)
        self.num_uniques = int(num_uniques)
        self.load_config()
        self.load_rules()

        self.problem = plp.LpProblem("MMA", plp.LpMaximize)

        projection_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["projection_path"]),
        )
        self.load_projections(projection_path)

        player_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["player_path"]),
        )
        self.load_player_ids(player_path)

    # Load config from file
    def load_config(self):
        with open(
            os.path.join(os.path.dirname(__file__), "../config.json"),
            encoding="utf-8-sig",
        ) as json_file:
            self.config = json.load(json_file)

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                name = row["name"].replace("-", "#")
                game_info = row["game info"].split(" ")
                matchup = game_info[0]
                fighters = matchup.split("@")
                fighter1 = fighters[0]
                fighter2 = fighters[1]
                opponent = fighter1 if fighter1 not in name else fighter2
                self.player_dict[name]["ID"] = row["id"]
                self.player_dict[name]["Opponent"] = opponent

    def load_rules(self):
        self.at_most = self.config["at_most"]
        self.at_least = self.config["at_least"]
        self.exactly = self.config["exactly"]
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])
        self.allow_fighters_from_same_fight = self.config[
            "allow_fighters_from_same_fight"
        ]

    def lower_first(self, iterator):
        return itertools.chain([next(iterator).lower()], iterator)

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                player_name = row["name"].replace("-", "#")
                if float(row["projection"]) < self.projection_minimum:
                    continue
                self.player_dict[player_name] = {
                    "Fpts": float(row["fight projection"]),
                    "ID": -1,
                    "Salary": int(row["salary"].replace(",", "")),
                    "Name": row["name"],
                    "Opponent": None,
                    "Ownership": (
                        float(row["ownership"] if "ownership" in row else 0.0)
                    ),
                    "Win%": float(row["winodds"] if "winodds" in row else 0.0),
                    "KO Win%": float(row["ko win"] if "ko win" in row else 0.0),
                    "Pace": float(row["pace"] if "pace" in row else 0.0),
                    "Avg Length": float(
                        row["avg length"] if "avg length" in row else 0.0
                    ),
                    "> 100fpts %": float(
                        row[">100 fpts%"] if ">100 fpts%" in row else 0.0
                    ),
                }
                self.fighter_list.append(player_name)

    def optimize(self):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot.
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        lp_variables = {
            player: plp.LpVariable(player, cat="Binary")
            for player, _ in self.player_dict.items()
        }

        # set the objective - maximize fpts & set randomness amount from config
        self.problem += (
            plp.lpSum(
                (
                    self.player_dict[player]["Fpts"]
                    + (self.player_dict[player]["Fpts"] * self.randomness_amount / 100)
                )
                * lp_variables[player]
                for player in self.player_dict
            ),
            "Objective",
        )
        # Set the salary constraints
        max_salary = 50000
        min_salary = 49000

        # Set min salary if in config
        if (
            "min_lineup_salary" in self.config
            and self.config["min_lineup_salary"] is not None
        ):
            min_salary = self.config["min_lineup_salary"]

        self.problem += (
            plp.lpSum(
                self.player_dict[player]["Salary"] * lp_variables[player]
                for player in self.player_dict
            )
            <= max_salary
        ), "Max Salary"
        self.problem += (
            plp.lpSum(
                self.player_dict[player]["Salary"] * lp_variables[player]
                for player in self.player_dict
            )
            >= min_salary
        ), "Min Salary"

        # Address limit rules if any
        for limit, groups in self.at_least.items():
            for group in groups:
                self.problem += (
                    plp.lpSum(
                        lp_variables[player.replace("-", "#")] for player in group
                    )
                    >= int(limit),
                    f"At least {limit} of {groups}",
                )

        for limit, groups in self.at_most.items():
            for group in groups:
                self.problem += (
                    plp.lpSum(
                        lp_variables[player.replace("-", "#")] for player in group
                    )
                    <= int(limit),
                    f"At most {limit} of {groups}",
                )

        for limit, groups in self.exactly.items():
            for group in groups:
                self.problem += (
                    plp.lpSum(
                        lp_variables[player.replace("-", "#")] for player in group
                    )
                    == int(limit),
                    f"Exactly {limit} of {groups}",
                )

        # Need 6 fighters. pretty easy.
        self.problem += (
            plp.lpSum(lp_variables[player] for player in self.player_dict) == 6
        ), "6 Fighters"

        if not self.allow_fighters_from_same_fight:
            seen_fighters = {}
            for fighter in self.fighter_list:
                opponent_last_name = self.player_dict[fighter]["Opponent"]
                opponent = [k for k in self.player_dict if opponent_last_name in k][0]

                if fighter in seen_fighters:
                    continue

                if opponent in seen_fighters:
                    continue

                self.problem += (
                    plp.lpSum(lp_variables[fighter] + lp_variables[opponent]) <= 1
                ), f"No fighter from same fight {fighter} vs {opponent}"

        # Crunch!
        for i in range(self.num_lineups):
            try:
                self.problem.solve(plp.PULP_CBC_CMD(msg=0))
            except plp.PulpSolverError:
                print(
                    "Infeasibility reached - only generated {} lineups out of {}. Continuing with export.".format(
                        len(self.num_lineups), self.num_lineups
                    )
                )

            if i % 100 == 0:
                print(i)

            # Get the lineup and add it to our list
            players = [
                player for player in lp_variables if lp_variables[player].varValue != 0
            ]

            self.lineups.append(players)

            # Ensure this lineup isn't picked again
            self.problem += (
                plp.lpSum(lp_variables[player] for player in players)
                <= len(players) - self.num_uniques,
                f"Lineup {i}",
            )

            # Set a new random fpts projection within their distribution
            if self.randomness_amount != 0:
                self.problem += (
                    plp.lpSum(
                        (
                            self.player_dict[player]["Fpts"]
                            + (
                                self.player_dict[player]["Fpts"]
                                * self.randomness_amount
                                / 100
                            )
                        )
                        * lp_variables[player]
                        for player in self.player_dict
                    ),
                    "Objective",
                )

    def output(self):
        print("Lineups done generating. Outputting.")

        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_optimal_lineups_{}.csv".format(
                self.site, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            ),
        )
        with open(out_path, "w") as f:
            f.write(
                "FIGHTER,FIGHTER,FIGHTER,FIGHTER,FIGHTER,FIGHTER,Salary,Projection,Own. Sum, Own. Prod,>100 Fpts% Sum, >100Fpts% Prod, Win% Sum, Win% Prod, KO Win% Sum, KO Win% Prod, Avg Pace\n"
            )
            for x in self.lineups:
                salary = sum(self.player_dict[player]["Salary"] for player in x)
                fpts_p = sum(self.player_dict[player]["Fpts"] for player in x)
                own_p = np.prod(
                    [self.player_dict[player]["Ownership"] / 100.0 for player in x]
                )
                own_s = sum([self.player_dict[player]["Ownership"] for player in x])
                grt100_p = np.prod(
                    [self.player_dict[player]["> 100fpts %"] / 100.0 for player in x]
                )
                grt100_s = sum(
                    [self.player_dict[player]["> 100fpts %"] for player in x]
                )
                win_p = np.prod(
                    [self.player_dict[player]["Win%"] / 100.0 for player in x]
                )
                win_s = sum([self.player_dict[player]["Win%"] for player in x])
                ko_p = np.prod(
                    [self.player_dict[player]["KO Win%"] / 100.0 for player in x]
                )
                ko_s = sum([self.player_dict[player]["KO Win%"] for player in x])
                avg_pace = sum([self.player_dict[player]["Pace"] for player in x]) / 6

                lineup_str = "{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{},{},{},{},{},{},{},{}".format(
                    self.player_dict[x[0]]["Name"],
                    self.player_dict[x[0]]["ID"],
                    self.player_dict[x[1]]["Name"],
                    self.player_dict[x[1]]["ID"],
                    self.player_dict[x[2]]["Name"],
                    self.player_dict[x[2]]["ID"],
                    self.player_dict[x[3]]["Name"],
                    self.player_dict[x[3]]["ID"],
                    self.player_dict[x[4]]["Name"],
                    self.player_dict[x[4]]["ID"],
                    self.player_dict[x[5]]["Name"],
                    self.player_dict[x[5]]["ID"],
                    salary,
                    round(fpts_p, 2),
                    own_s,
                    own_p,
                    grt100_s,
                    grt100_p,
                    win_s,
                    win_p,
                    ko_s,
                    ko_p,
                    round(avg_pace, 2),
                )
                f.write("%s\n" % lineup_str)

        print("Output done.")
