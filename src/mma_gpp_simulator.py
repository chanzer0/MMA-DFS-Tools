import csv
import json
import math
import os
import random
import time
import numpy as np
import pulp as plp
import multiprocessing as mp
import pandas as pd
import numba as nb 
import itertools
import pickle
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns 
import re
import datetime
import scipy.stats as stats
from numpy.random import default_rng

@nb.jit(nopython=True)  # nopython mode ensures the function is fully optimized
def salary_boost(salary, max_salary):
    return (salary / max_salary) ** 2

class MMA_GPP_Simulator:
    config = None
    player_dict = {}
    field_lineups = {}
    gen_lineup_list = []
    roster_construction = []
    salary = None
    optimal_score = None
    field_size = None
    team_list = []
    num_iterations = None
    site = None
    payout_structure = {}
    use_contest_data = False
    cut_event = False
    entry_fee = None
    use_lineup_input = None
    projection_minimum = 15
    randomness_amount = 100
    min_lineup_salary = 48000
    max_pct_off_optimal = 0.4
    seen_lineups = {}
    seen_lineups_ix = {}
    game_info = {}
    matchups = {}
    allow_opps = 0

    def __init__(
        self,
        site,
        field_size,
        num_iterations,
        use_contest_data,
        use_lineup_input,
        match_lineup_input_to_field_size,
    ):
        self.site = site
        self.use_lineup_input = use_lineup_input
        self.match_lineup_input_to_field_size = match_lineup_input_to_field_size
        self.load_config()
        self.load_rules()
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

        if site == "dk":
            self.roster_construction = ["F", "F", "F", "F", "F", "F"]
            self.salary = 50000

        self.use_contest_data = use_contest_data
        if use_contest_data:
            contest_path = os.path.join(
                os.path.dirname(__file__),
                "../{}_data/{}".format(site, self.config["contest_structure_path"]),
            )
            self.load_contest_data(contest_path)
            print("Contest payout structure loaded.")
        else:
            self.field_size = int(field_size)
            self.payout_structure = {0: 0.0}
            self.entry_fee = 0
        self.num_iterations = int(num_iterations)
        self.get_optimal()
        if self.use_lineup_input:
            self.load_lineups_from_file()
        if self.match_lineup_input_to_field_size or len(self.field_lineups) == 0:
            self.generate_field_lineups()

    def load_rules(self):
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])
        self.min_lineup_salary = int(self.config["min_lineup_salary"])
        self.max_pct_off_optimal = float(self.config["max_pct_off_optimal"])

    # In order to make reasonable tournament lineups, we want to be close enough to the optimal that
    # a person could realistically land on this lineup. Skeleton here is taken from base `nba_optimizer.py`
    def get_optimal(self):
        problem = plp.LpProblem("MMA", plp.LpMaximize)
        lp_variables = {
            player: plp.LpVariable(player, cat="Binary")
            for player, _ in self.player_dict.items()
        }

        # set the objective - maximize fpts
        problem += (
            plp.lpSum(
                self.player_dict[player]["FieldFpts"] * lp_variables[player]
                for player in self.player_dict
            ),
            "Objective",
        )

        # Set the salary constraints
        problem += (
            plp.lpSum(
                self.player_dict[player]["Salary"] * lp_variables[player]
                for player in self.player_dict
            )
            <= self.salary
        )

        if self.site == "dk":
            # Can only roster 6 total players
            problem += (
                plp.lpSum(lp_variables[player] for player in self.player_dict) == 6
            )

        # Crunch!
        try:
            problem.solve(plp.PULP_CBC_CMD(msg=0))
        except plp.PulpSolverError:
            print(
                "Infeasibility reached - we failed to generate an optimal lineup. Please check all files are present and formatted correctly. Otherwise, submit a ticket on the github."
            )

        score = str(problem.objective)
        for v in problem.variables():
            score = score.replace(v.name, str(v.varValue))

        self.optimal_score = eval(score)

    @staticmethod
    def extract_matchup_time(game_string):
        # Extract the matchup, date, and time
        match = re.match(
            r"(\w{2,4}@\w{2,4}) (\d{2}/\d{2}/\d{4}) (\d{2}:\d{2}[APM]{2} ET)",
            game_string,
        )

        if match:
            matchup, date, time = match.groups()
            # Convert 12-hour time format to 24-hour format
            time_obj = datetime.datetime.strptime(time, "%I:%M%p ET")
            # Convert the date string to datetime.date
            date_obj = datetime.datetime.strptime(date, "%m/%d/%Y").date()
            # Combine date and time to get a full datetime object
            datetime_obj = datetime.datetime.combine(date_obj, time_obj.time())
            return matchup, datetime_obj
        return None

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                name_key = "Name" if self.site == "dk" else "Nickname"
                player_name = row[name_key].replace("-", "#").lower()
                team_key = "TeamAbbrev" if self.site == "dk" else "Team"
                team = row[team_key]
                # This will capture something like "LAC@GSW 01/01/2023 08:00PM ET"
                game_info_key = "Game Info" if self.site == "dk" else "game"
                game_info_str = row[game_info_key]
                # Extract matchup and datetime information
                match = re.search(r"(\w+)@(\w+) (\d{2}/\d{2}/\d{4} \d{2}:\d{2}[AP]M ET)", row['Game Info'])
                if match:
                    player1_team, player2_team, datetime_str = match.groups()

                    # Create a matchup key based on the team (or player) abbreviations
                    matchup_key = (player1_team, player2_team)

                    # Populate the matchups dictionary with player details
                    if matchup_key not in self.matchups:
                        self.matchups[matchup_key] = []

                    # Determine the opponent based on the matchup key and the player's team
                    opponent = player2_team if player1_team == team else player1_team

                    # Before appending, ensure the player's dictionary has the 'Opp' field updated
                    if player_name in self.player_dict:
                        self.player_dict[player_name]['Opp'] = opponent

                    # Add the player's details to the matchup
                    self.matchups[matchup_key].append(self.player_dict[player_name])

                if player_name in self.player_dict:
                    if self.site == "dk":
                        self.player_dict[player_name]["ID"] = int(row["ID"])
                        self.player_dict[player_name]["UniqueKey"] = int(row["ID"])
                        self.player_dict[player_name]["Opp"] = team
                    else:
                        self.player_dict[player_name]["ID"] = row["Id"]
                else:
                    print(player_name + " not found in player dict")
        #print(self.matchups)
                
    def load_contest_data(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if self.field_size is None:
                    self.field_size = int(row["Field Size"])
                if self.entry_fee is None:
                    self.entry_fee = float(row["Entry Fee"])
                # multi-position payouts
                if "-" in row["Place"]:
                    indices = row["Place"].split("-")
                    # print(indices)
                    # have to add 1 to range to get it to generate value for everything
                    for i in range(int(indices[0]), int(indices[1]) + 1):
                        # print(i)
                        # Where I'm from, we 0 index things. Thus, -1 since Payout starts at 1st place
                        if i >= self.field_size:
                            break
                        self.payout_structure[i - 1] = float(
                            row["Payout"].split(".")[0].replace(",", "")
                        )
                # single-position payouts
                else:
                    if int(row["Place"]) >= self.field_size:
                        break
                    self.payout_structure[int(row["Place"]) - 1] = float(
                        row["Payout"].split(".")[0].replace(",", "")
                    )
        # print(self.payout_structure)
    
    def lower_first(self, iterator):
        return itertools.chain([next(iterator).lower()], iterator)

    # Load config from file
    def load_config(self):
        with open(
            os.path.join(os.path.dirname(__file__), "../config.json"),
            encoding="utf-8-sig",
        ) as json_file:
            self.config = json.load(json_file)

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                player_name = row["name"].replace("-", "#").lower()
                fpts = float(row["fpts"])
                if fpts < self.projection_minimum:
                    continue
                if "fieldfpts" in row:
                    if row["fieldfpts"] == "":
                        fieldFpts = fpts
                    else:
                        fieldFpts = float(row["fieldfpts"])
                else:
                    fieldFpts = fpts
                if row["salary"]:
                    sal = int(row["salary"].replace(",", ""))
                if 'ko prob' in row:
                    koprob = float(row['ko prob'])/100
                else:
                    koprob = 0
                if 'win prob' in row:
                    winprob = float(row['win prob'])/100
                else:
                    winprob = 0
                own = float(row["own%"].replace("%", ""))
                if own == 0:
                    own = 0.1         
                self.player_dict[player_name] = {
                    "Fpts": fpts,
                    "Position": ['F'],
                    "ID": 0,
                    "Salary": sal,
                    "Ownership": own,
                    "In Lineup": False,
                    "Name": row['name'],
                    "FieldFpts": fieldFpts,
                    "KoProb": koprob,
                    "WinProb": winprob,
                    "UniqueKey": 0
                }

    def extract_id(self,cell_value):
        if "(" in cell_value and ")" in cell_value:
            return cell_value.split("(")[1].replace(")", "")
        elif ":" in cell_value:
            return cell_value.split(":")[0]
        else:
            return cell_value

    def load_lineups_from_file(self):
        print("loading lineups")
        i = 0
        path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(self.site, "tournament_lineups.csv"),
        )
        with open(path) as file:
            reader = pd.read_csv(file)
            lineup = []
            bad_lus = []
            bad_players = []
            
            j = 0
            for i, row in reader.iterrows():
                if i == self.field_size:
                    break
                lineup = [
                    int(self.extract_id(str(row[q])))
                    for q in range(len(self.roster_construction))
                ]
                lu_names = []
                # storing if this lineup was made by an optimizer or with the generation process in this script
                error = False
                for l in lineup:
                    ids = [self.player_dict[k]["ID"] for k in self.player_dict]
                    if l not in ids:
                        print("player id {} in lineup {} not found in player dict".format(l, i))
                        #if l in self.id_name_dict:
                        #    print(self.id_name_dict[l])
                        bad_players.append(l)
                        error = True
                    else:
                        for k in self.player_dict:
                            if self.player_dict[k]["ID"] == l:
                                lu_names.append(k)
                lu_names = lineup
                if len(lineup) < len(self.roster_construction):
                    print("lineup {} doesn't match roster construction size".format(i))
                    continue
                # storing if this lineup was made by an optimizer or with the generation process in this script
                error = False
                if len(lu_names) != len(self.roster_construction):
                    error = True
                if not error:
                    lineup_list = sorted(lineup)           
                    lineup_set = frozenset(lineup_list)

                    # Keeping track of lineup duplication counts
                    if lineup_set in self.seen_lineups:
                        self.seen_lineups[lineup_set] += 1
                    else:
                        self.field_lineups[j] = {
                                "Lineup": lu_names,
                                "Wins": 0,
                                "Top1Percent": 0,
                                "ROI": 0,
                                "Cashes": 0,
                                "Type": "opto",
                                "Count": 1
                        }
                        # Add to seen_lineups and seen_lineups_ix
                        self.seen_lineups[lineup_set] = 1
                        self.seen_lineups_ix[lineup_set] = j
                        j += 1
        print("loaded {} lineups".format(j))
        # print(self.field_lineups)

    @staticmethod
    def generate_lineups(
        lu_num,
        names,
        in_lineup,
        pos_matrix,
        ownership,
        salary_floor,
        salary_ceiling,
        optimal_score,
        salaries,
        projections,
        max_pct_off_optimal,
        opponents_dict
    ):
        # new random seed for each lineup (without this there is a ton of dupes)
        rng = np.random.Generator(np.random.PCG64())
        min_salary = np.quantile(salaries, 0.3)
        lus = {}
        # make sure nobody is already showing up in a lineup
        if sum(in_lineup) != 0:
            in_lineup.fill(0)
        reject = True
        while reject:
            salary = 0
            proj = 0
            if sum(in_lineup) != 0:
                in_lineup.fill(0)
            lineup = []
            q = 0
            for pos in pos_matrix.T:
                # calculate difference between current lineup salary and salary ceiling
                if q < 5:
                    salary_diff = salary_ceiling - (salary + min_salary)
                else:
                    salary_diff = salary_ceiling - salary
                # check for players eligible for the position and make sure they arent in a lineup and the player's salary is less than or equal to salary_diff, returns a list of indices of available player
                valid_player_indices = np.where((pos > 0) & (in_lineup == 0) & (salaries <= salary_diff))[0]

                # Convert current lineup to a set of integers for faster lookup
                current_lineup_set = set(int(name) for name in lineup)

                # Filter valid player indices based on opponent constraints
                valid_players = [idx for idx in valid_player_indices if not set(opponents_dict.get(names[idx], [])).intersection(current_lineup_set)]

                if len(valid_players) == 0:
                    # Force restart if no valid players are found
                    reject = True
                    break    
                # grab names of players eligible
                plyr_list = names[valid_players]
                # create np array of probability of being seelcted based on ownership and who is eligible at the position
                prob_list = ownership[valid_players]
                prob_list = prob_list / prob_list.sum()  # normalize to ensure it sums to 1
                if q == 5:
                    boosted_salaries = np.array([salary_boost(s, salary_ceiling) for s in salaries[valid_players]])
                    boosted_probabilities = prob_list * boosted_salaries
                    boosted_probabilities /= boosted_probabilities.sum()  # normalize to ensure it sums to 1
                try:
                    if q == 5:
                        choice = rng.choice(plyr_list, p=boosted_probabilities)
                    else:
                        choice = rng.choice(plyr_list, p=prob_list)
                except:
                    # if remaining_salary <= np.min(salaries):
                    #     reject_counters["salary_too_high"] += 1
                    # else:
                    #     reject_counters["salary_too_low"]
                    salary = 0
                    proj = 0
                    lineup = []
                    in_lineup.fill(0)  # Reset the in_lineup array
                    k = 0  # Reset the player index
                    continue  # Skip to the next iteration of the while loop
                choice_idx = np.where(names == choice)[0]
                lineup.append(choice)
                in_lineup[choice_idx] = 1
                salary += salaries[choice_idx]
                proj += projections[choice_idx]
                q+=1
            # Must have a reasonable salary
            if salary >= salary_floor and salary <= salary_ceiling and len(lineup) == 6:
                # Must have a reasonable projection (within 60% of optimal) **people make a lot of bad lineups
                reasonable_projection = optimal_score - (
                    max_pct_off_optimal * optimal_score
                )
                if proj >= reasonable_projection:
                    reject = False
                    lu = {
                        "Lineup": lineup,
                        "Wins": 0,
                        "Top1Percent": 0,
                        "ROI": 0,
                        "Cashes": 0,
                        "Type": "generated",
                        "Count": 0
                    }
        return lu

    def generate_field_lineups(self):
        diff = self.field_size - len(self.field_lineups)
        if diff <= 0:
            print(
                "supplied lineups >= contest field size. only retrieving the first "
                + str(self.field_size)
                + " lineups"
            )
        else:
            print("Generating " + str(diff) + " lineups.")
            ids = []
            ownership = []
            salaries = []
            projections = []
            positions = []
            # put def first to make it easier to avoid overlap
            for k in self.player_dict.keys():
                ids.append(self.player_dict[k]["UniqueKey"])
                ownership.append(self.player_dict[k]["Ownership"])
                salaries.append(self.player_dict[k]["Salary"])
                if self.player_dict[k]["FieldFpts"] >= self.projection_minimum:
                    projections.append(self.player_dict[k]["FieldFpts"])
                else:
                    projections.append(0)
                pos_list = []
                for pos in self.roster_construction:
                    if pos in self.player_dict[k]["Position"]:
                        pos_list.append(1)
                    else:
                        pos_list.append(0)
                positions.append(np.array(pos_list))
            in_lineup = np.zeros(shape=len(ids))
            ownership = np.array(ownership)
            salaries = np.array(salaries)
            projections = np.array(projections)
            pos_matrix = np.array(positions)
            ids = np.array(ids)
            optimal_score = self.optimal_score
            salary_floor = (
                self.min_lineup_salary
            )  # anecdotally made the most sense when looking at previous contests
            salary_ceiling = self.salary
            max_pct_off_optimal = self.max_pct_off_optimal
            problems = []
            opponents_dict = {}
            if self.allow_opps == 0:
                for matchup in self.matchups.values():
                    for player in matchup:
                        opponents = [op["UniqueKey"] for op in matchup if op["UniqueKey"] != player["UniqueKey"]]
                        opponents_dict[player["UniqueKey"]] = opponents
                #print(opponents_dict)
            # creating tuples of the above np arrays plus which lineup number we are going to create
            for i in range(diff):
                lu_tuple = (
                    i,
                    ids,
                    in_lineup,
                    pos_matrix,
                    ownership,
                    salary_floor,
                    salary_ceiling,
                    optimal_score,
                    salaries,
                    projections,
                    max_pct_off_optimal,
                    opponents_dict
                )
                problems.append(lu_tuple)
            start_time = time.time()
            with mp.Pool() as pool:
                output = pool.starmap(self.generate_lineups, problems)
                print(
                    "number of running processes =",
                    pool.__dict__["_processes"]
                    if (pool.__dict__["_state"]).upper() == "RUN"
                    else None,
                )
                pool.close()
                pool.join()
            print("pool closed")
            self.update_field_lineups(output, diff)
            msg = str(diff) + " field lineups successfully generated. " + str(len(self.field_lineups.keys())) + " uniques."
            end_time = time.time()
            #self.simDoc.update({'jobProgressLog': ArrayUnion([msg])})
            print("lineups took " + str(end_time - start_time) + " seconds")
            # print(self.field_lineups)

        # print(self.field_lineups)
    def update_field_lineups(self, output, diff):
        if len(self.field_lineups) == 0:
            new_keys = list(range(0, self.field_size))
        else:
            new_keys = list(range(max(self.field_lineups.keys()) + 1, max(self.field_lineups.keys()) + 1 + diff))
        nk = new_keys[0]
        for i, o in enumerate(output):
            #print(o.values())
            lineup_list = sorted(o['Lineup'])
            lineup_set = frozenset(lineup_list)
            #print(lineup_set)

            # Keeping track of lineup duplication counts
            if lineup_set in self.seen_lineups:
                self.seen_lineups[lineup_set] += 1
                        
                # Increase the count in field_lineups using the index stored in seen_lineups_ix
                self.field_lineups[self.seen_lineups_ix[lineup_set]]["Count"] += 1
            else:
                self.seen_lineups[lineup_set] = 1
                
                if nk in self.field_lineups.keys():
                    print("bad lineups dict, please check dk_data files")
                else:
                    # Convert dict_values to a dictionary before assignment
                    lineup_data = dict(o)
                    lineup_data['Lineup'] = lineup_list
                    lineup_data['Count'] += self.seen_lineups[lineup_set]

                    # Now assign the dictionary to the field_lineups
                    self.field_lineups[nk] = lineup_data                 
                    # Store the new nk in seen_lineups_ix for quick access in the future
                    self.seen_lineups_ix[lineup_set] = nk
                    nk += 1

    @staticmethod
    def worker_function(matchup_data, num_iterations, plot_folder, kmeans_model, scaler, gmm_models):
        # Unpack matchup_data which contains the fighters' data
        #np.random.seed(42)
        rng = default_rng()
        fighters_data = matchup_data

        def plot_simulation_results(fighter1_samples, fighter2_samples, fighter1_name, fighter2_name, plot_folder):
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(10, 6))

            # Plot histograms
            sns.histplot(fighter1_samples, color="blue", label=fighter1_name, kde=True, stat="density", bins=50, alpha=0.5)
            sns.histplot(fighter2_samples, color="red", label=fighter2_name, kde=True, stat="density", bins=50, alpha=0.5)

            # Compute and display correlation
            correlation = np.corrcoef(fighter1_samples, fighter2_samples)[0, 1]
            
            # Adjust text position and style for better visibility
            plt.text(0.05, 0.95, f"Correlation: {correlation:.2f}", transform=plt.gca().transAxes, fontsize=12, fontweight='bold', color='black', verticalalignment='top')

            plt.legend()
            plt.title("Simulated Fantasy Points Distributions")
            plt.xlabel("Fantasy Points")
            plt.ylabel("Density")

            # Save plot
            plot_path = os.path.join(plot_folder, f'{fighter1_name}_vs_{fighter2_name}_simulation.png')
            plt.savefig(plot_path)
            plt.close()

        def process_fighter(fighter_data, scaled_data, gmm_models, wins, iteration):
            player_cluster = kmeans_model.predict(scaled_data)[0]
            gmm = gmm_models[player_cluster]
            
            loss_component, win_component = (0, 1) if gmm.means_[0, 0] < gmm.means_[1, 0] else (1, 0)
            component = win_component if wins else loss_component

            sample = rng.multivariate_normal(gmm.means_[component], gmm.covariances_[component])[0]
            return sample

        if len(fighters_data) == 2:
            fighter1_data, fighter2_data = fighters_data

            # Log initial data for fighters
            #print(f"Initial data - Fighter 1: {fighter1_data['Name']}, WinProb: {fighter1_data['WinProb']}, KoProb: {fighter1_data['KoProb']}")
            #print(f"Initial data - Fighter 2: {fighter2_data['Name']}, WinProb: {fighter2_data['WinProb']}, KoProb: {fighter2_data['KoProb']}")

            # Preprocess win and KO odds outside the loop
            fighter1_preprocessed = np.array([[fighter1_data['WinProb'], fighter1_data['KoProb']]])
            fighter2_preprocessed = np.array([[fighter2_data['WinProb'], fighter2_data['KoProb']]])
            scaled_fighter1_data = scaler.transform(fighter1_preprocessed)
            scaled_fighter2_data = scaler.transform(fighter2_preprocessed)

            # Prepare arrays for simulation results
            fighter1_samples = np.zeros(num_iterations)
            fighter2_samples = np.zeros(num_iterations)

            for i in range(num_iterations):
                fighter1_wins = rng.random() < fighter1_data['WinProb']

                fighter1_cluster = process_fighter(fighter1_data, scaled_fighter1_data, gmm_models, fighter1_wins,i)
                fighter1_samples[i] = fighter1_cluster

                fighter2_cluster = process_fighter(fighter2_data, scaled_fighter2_data, gmm_models, not fighter1_wins,i)
                fighter2_samples[i] = fighter2_cluster

            # Log simulation stats
            #print(f"Simulation stats for {fighter1_data['Name']} - Median: {np.median(fighter1_samples)}, StdDev: {np.std(fighter1_samples)}")
            #print(f"Simulation stats for {fighter2_data['Name']} - Median: {np.median(fighter2_samples)}, StdDev: {np.std(fighter2_samples)}")

            # Adjust samples based on the desired median shift
            median_shift_fighter1 = fighter1_data['Fpts'] - np.median(fighter1_samples)
            fighter1_samples += median_shift_fighter1
            median_shift_fighter2 = fighter2_data['Fpts'] - np.median(fighter2_samples)
            fighter2_samples += median_shift_fighter2

            # Plotting and saving the results for both fighters
            #plot_simulation_results(fighter1_samples, fighter2_samples, fighter1_data['Name'], fighter2_data['Name'], plot_folder)

            # Return simulation results
       # print(f"Adjusted simulation for {fighter1_data['Name']} vs {fighter2_data['Name']} complete.")
        return {fighter1_data['ID']: fighter1_samples, fighter2_data['ID']: fighter2_samples}

    @staticmethod
    @nb.jit(nopython=True)
    def calculate_payouts(args):
        (
            ranks,
            payout_array,
            entry_fee,
            field_lineup_keys,
            use_contest_data,
            field_lineups_count,
        ) = args
        num_lineups = len(field_lineup_keys)
        combined_result_array = np.zeros(num_lineups)

        payout_cumsum = np.cumsum(payout_array)

        for r in range(ranks.shape[1]):
            ranks_in_sim = ranks[:, r]
            payout_index = 0
            for lineup_index in ranks_in_sim:
                lineup_count = field_lineups_count[lineup_index]
                prize_for_lineup = (
                    (
                        payout_cumsum[payout_index + lineup_count - 1]
                        - payout_cumsum[payout_index - 1]
                    )
                    / lineup_count
                    if payout_index != 0
                    else payout_cumsum[payout_index + lineup_count - 1] / lineup_count
                )
                combined_result_array[lineup_index] += prize_for_lineup
                payout_index += lineup_count
        return combined_result_array
       
    def run_tournament_simulation(self):
        print("Running " + str(self.num_iterations) + " simulations")
        start_time = time.time()
        plot_folder = "simulation_plots"
        # Create a directory for plots if it does not exist
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        kmeans_5 = pickle.load(open("src/cluster_data/kmeans.pkl", "rb"))
        scaler = pickle.load(open("src/cluster_data/scaler.pkl", "rb"))
        gmm_models = pickle.load(open("src/cluster_data/gmm_models.pkl", "rb"))
        # Adding prints for GMM components means, covariances, and selected component
       # for k,v in gmm_models.items():
        #    print(f"Cluster: {k}, GMM Means: {v.means_}, GMM Covariances: {v.covariances_}")
     
        #print(self.matchups)
        # Prepare the matchups data for simulation
        matchups_data = [
            (list(self.matchups[m]), self.num_iterations, plot_folder, kmeans_5, scaler, gmm_models)
            for m in self.matchups
        ]
        #print(matchups_data[0])
        # Create a pool of workers and distribute the work
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.starmap(self.worker_function, matchups_data)

        # Process results and store them in temp_fpts_dict
        temp_fpts_dict = {}
        for result in results:
            for fighter, samples in result.items():
                # Calculate statistical metrics
                median = np.median(samples)
                variance = np.var(samples)
                stddev = np.std(samples)
                mean = np.mean(samples)
                max = np.max(samples)
                sample_min = np.min(samples)
                percentile_25 = np.percentile(samples, 25)
                percentile_75 = np.percentile(samples, 75)

                # Store samples in the dict
                temp_fpts_dict[fighter] = samples

                # Print statistical information
                #print(f"Stats for {fighter}: Median={median}, Variance={variance}, StdDev={stddev}, Mean={mean}, Max={max}, Min={sample_min}, 25th Percentile={percentile_25}, 75th Percentile={percentile_75}")
        # generate arrays for every sim result for each player in the lineup and sum
        fpts_array = np.zeros(shape=(len(self.field_lineups), self.num_iterations))
        # converting payout structure into an np friendly format, could probably just do this in the load contest function
        # print(self.field_lineups)
        # print(temp_fpts_dict)
        # print(payout_array)
        # print(self.player_dict[('patrick mahomes', 'FLEX', 'KC')])
        field_lineups_count = np.array(
            [self.field_lineups[idx]["Count"] for idx in self.field_lineups.keys()]
        )

        for index, values in self.field_lineups.items():
            try:
                fpts_sim = sum([temp_fpts_dict[player] for player in values["Lineup"]])
            except KeyError:
                for player in values["Lineup"]:
                    if player not in temp_fpts_dict.keys():
                        print(player)
                        # for k,v in self.player_dict.items():
                        # if v['ID'] == player:
                        #        print(k,v)
                # print('cant find player in sim dict', values["Lineup"], temp_fpts_dict.keys())
            # store lineup fpts sum in 2d np array where index (row) corresponds to index of field_lineups and columns are the fpts from each sim
            fpts_array[index] = fpts_sim

        fpts_array = fpts_array.astype(np.float16)
        # ranks = np.argsort(fpts_array, axis=0)[::-1].astype(np.uint16)
        ranks = np.argsort(-fpts_array, axis=0).astype(np.uint32)

        # count wins, top 10s vectorized
        wins, win_counts = np.unique(ranks[0, :], return_counts=True)
        cashes, cash_counts = np.unique(ranks[0:len(list(self.payout_structure.values()))], return_counts=True)

        top1pct, top1pct_counts = np.unique(
            ranks[0 : math.ceil(0.01 * len(self.field_lineups)), :], return_counts=True
        )

        payout_array = np.array(list(self.payout_structure.values()))
        # subtract entry fee
        payout_array = payout_array - self.entry_fee
        l_array = np.full(
            shape=self.field_size - len(payout_array), fill_value=-self.entry_fee
        )
        payout_array = np.concatenate((payout_array, l_array))
        field_lineups_keys_array = np.array(list(self.field_lineups.keys()))

        # Adjusted ROI calculation
        # print(field_lineups_count.shape, payout_array.shape, ranks.shape, fpts_array.shape)

        # Split the simulation indices into chunks
        field_lineups_keys_array = np.array(list(self.field_lineups.keys()))

        chunk_size = self.num_iterations // 16  # Adjust chunk size as needed
        simulation_chunks = [
            (
                ranks[:, i : min(i + chunk_size, self.num_iterations)].copy(),
                payout_array,
                self.entry_fee,
                field_lineups_keys_array,
                self.use_contest_data,
                field_lineups_count,
            )  # Adding field_lineups_count here
            for i in range(0, self.num_iterations, chunk_size)
        ]

        # Use the pool to process the chunks in parallel
        with mp.Pool() as pool:
            results = pool.map(self.calculate_payouts, simulation_chunks)

        combined_result_array = np.sum(results, axis=0)

        total_sum = 0
        index_to_key = list(self.field_lineups.keys())
        for idx, roi in enumerate(combined_result_array):
            lineup_key = index_to_key[idx]
            lineup_count = self.field_lineups[lineup_key][
                "Count"
            ]  # Assuming "Count" holds the count of the lineups
            total_sum += roi * lineup_count
            self.field_lineups[lineup_key]["ROI"] += roi

        for idx in self.field_lineups.keys():
            if idx in wins:
                self.field_lineups[idx]["Wins"] += win_counts[np.where(wins == idx)][0]
            if idx in top1pct:
                self.field_lineups[idx]["Top1Percent"] += top1pct_counts[
                    np.where(top1pct == idx)
                ][0]
            if idx in cashes:
                self.field_lineups[idx]["Cashes"] += cash_counts[np.where(cashes == idx)][0]
        end_time = time.time()
        diff = end_time - start_time
        print(
            str(self.num_iterations)
            + " tournament simulations finished in "
            + str(diff)
            + "seconds. Outputting."
        )
        msg = str(self.num_iterations) + " Tournament simulations finished in " + str(round(diff, 2)) + " seconds. Outputting."

    def output(self):
        self.player_dict = {v['UniqueKey']: v for k, v in self.player_dict.items()}
        unique = {}
        tot_wins = 0
        tot_roi = 0
        for index, x in self.field_lineups.items():
            salary = sum(self.player_dict[player]["Salary"] for player in x["Lineup"])
            fpts_p = sum(self.player_dict[player]["Fpts"] for player in x["Lineup"])
            #ceil_p = sum(self.player_dict[player]["Ceiling"] for player in x["Lineup"])
            own_p = np.prod(
                [
                    self.player_dict[player]["Ownership"] / 100.0
                    for player in x["Lineup"]
                ]
            )
            win_p = round(x["Wins"] / self.num_iterations * 100, 2)
            top10_p = round(x["Top1Percent"] / self.num_iterations * 100, 2)
            cash_p = round(x["Cashes"] / self.num_iterations * 100, 2)
            lineup_names = [self.player_dict[player]["Name"] for player in x["Lineup"]]
            lu_type = x["Type"]
            simDupes = x['Count']
            tot_wins += x["Wins"]
            tot_roi += x["ROI"]
            if self.site == "dk":
                if self.use_contest_data:
                    roi_p = round((x["ROI"] / (simDupes *self.entry_fee *self.num_iterations)) * 100, 2)
                    roi_round = round(x["ROI"] / simDupes / self.num_iterations, 2)
                    lineup_str = "{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{}%,{}%,{}%,{}%,{},${},{},{}".format(
                        lineup_names[0].replace("#", "-"),
                        x["Lineup"][0],
                        lineup_names[1].replace("#", "-"),
                        x["Lineup"][1],
                        lineup_names[2].replace("#", "-"),
                        x["Lineup"][2],
                        lineup_names[3].replace("#", "-"),
                        x["Lineup"][3],
                        lineup_names[4].replace("#", "-"),
                        x["Lineup"][4],
                        lineup_names[5].replace("#", "-"),
                        x["Lineup"][5],
                        fpts_p,
                        #ceil_p,
                        salary,
                        win_p,
                        top10_p,
                        cash_p,
                        roi_p,
                        own_p,
                        roi_round,
                        lu_type,
                        simDupes,
                    )
                else:
                    lineup_str = "{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{}%,{}%,{},{}%,{},{}".format(
                        x["Lineup"][0].replace("#", "-"),
                        self.player_dict[x["Lineup"][0]]["ID"],
                        x["Lineup"][1].replace("#", "-"),
                        self.player_dict[x["Lineup"][1]]["ID"],
                        x["Lineup"][2].replace("#", "-"),
                        self.player_dict[x["Lineup"][2]]["ID"],
                        x["Lineup"][3].replace("#", "-"),
                        self.player_dict[x["Lineup"][3]]["ID"],
                        x["Lineup"][4].replace("#", "-"),
                        self.player_dict[x["Lineup"][4]]["ID"],
                        x["Lineup"][5].replace("#", "-"),
                        self.player_dict[x["Lineup"][5]]["ID"],
                        fpts_p,
                        #ceil_p,
                        salary,
                        win_p,
                        top10_p,
                        own_p,
                        cash_p,
                        lu_type,
                        simDupes
                    )
            unique[index] = lineup_str
        print(f'total wins {tot_wins}, total roi {tot_roi}')
        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_gpp_sim_lineups_{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations
            ),
        )
        with open(out_path, "w") as f:
            if self.site == "dk":
                if self.use_contest_data:
                    f.write(
                        "F,F,F,F,F,F,Fpts Proj,Salary,Win %,Top 1%,Cash%,ROI%,Proj. Own. Product, Avg. Return,Type,Simulated Duplicates,FptsMean,FptsStd,RanksMean,RanksStd\n"
                    )
            for fpts, lineup_str in unique.items():
                f.write("%s\n" % lineup_str)

        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_gpp_sim_player_exposure_{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations
            ),
        )
        # Initialize all player data
        unique_players = {player: {"Wins": 0, "Top1Percent": 0, "In": 0, "ROI": 0, "Cashes": 0} for player in self.player_dict}

        # Loop over all lineups and their outcomes once to aggregate player data
        for val in self.field_lineups.values():
            lineup_players = val["Lineup"]
            for player in lineup_players:
                unique_players[player]["In"] += 1
                unique_players[player]["ROI"] += val["ROI"]
                unique_players[player]["Cashes"] += val["Cashes"]
                
                # Only increment Wins and Top1Percent if the lineup has them
                if val['Wins'] > 0:
                    unique_players[player]["Wins"] += 1 / len(lineup_players)  # Distribute the win among the players in the lineup
                if val['Top1Percent'] > 0:
                    unique_players[player]["Top1Percent"] += 1 / len(lineup_players)  # Distribute the top 1% finish among the players in the lineup

        # Write the aggregated data to the output file
        with open(out_path, "w") as f:
            f.write("Player,Win%,Top1%,Cash%,Sim. Own%,Proj. Own%,Avg. Return\n")
            
            for player, data in unique_players.items():
                win_p = round(data["Wins"] / self.num_iterations * 100, 4)
                top10_p = round(data["Top1Percent"] / self.num_iterations * 100, 4)
                cash_p = round(data["Cashes"] / data["In"] / self.num_iterations * 100, 4)
                roi_p = round(data["ROI"] / data["In"] / self.num_iterations, 4)
                field_p = round(data["In"] / self.field_size * 100, 4)
                proj_own = self.player_dict[player]["Ownership"]*100
                
                f.write(
                    "{},{}%,{}%,{}%,{}%,{}%,${}\n".format(
                        self.player_dict[player]['Name'].replace("#", "-"),
                        win_p,
                        top10_p,
                        cash_p,
                        field_p,
                        proj_own,
                        roi_p,
                    )
                )