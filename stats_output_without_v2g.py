import numpy as np
from tqdm import tqdm
from pprint import pprint
import pandas as pd
import argparse

def get_input_args():
    """
    Returns input arguments for main file execution
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=500,
                        help='Number of episodes to run')
    parser.add_argument('--id_run', type=str, default='test_run',
                        help='id of run')
    parser.add_argument('--pen', type=float, default=0.1,
                        help='market penetration of evs')
    parser.add_argument('--avg_param', type=int, default=1,
                        help='if avg == 1, non-one avg and non-zero max are used')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='alpha for learning')
    parser.add_argument('--scale', type=int, default=1000,
                        help='scale')
    return parser.parse_args()

# Get args
n_episodes = get_input_args().n
id_run = get_input_args().id_run
pen = get_input_args().pen
avg = get_input_args().avg_param
alpha = get_input_args().alpha
scale = get_input_args().scale

# Get Alberta Average demand and prices
df = pd.read_csv('AESO_2020_demand_price.csv')
HE = []
end_index = df.shape[0] // (48 * 2) + 1
for day in range(1, end_index):
    for hour in range(1, (2 * 48) + 1):
        HE.append(hour)
df['HE'] = HE
df = df.drop(df.columns[[0, 2]], axis=1)
df = df.set_index('HE', drop=True)
df = df.groupby('HE', as_index=True).mean()
df_to_plot = df.drop(df.columns[[0]], axis=1)

alberta_avg_power_price = np.array(df.iloc[:, 0])
alberta_avg_demand = np.array(df.iloc[:, 1]) / scale

# https://open.alberta.ca/dataset/d6205817-b04b-4360-8bb0-79eaaecb9df9/
# resource/4a06c219-03d1-4027-9c1f-a383629ab3bc/download/trans-motorized-
# vehicle-registrations-select-municipalities-2020.pdf
total_cars_in_alberta = 100
ev_market_penetration = 0.1
charging_soc_addition_per_time_unit_per_ev = 0.15
discharging_soc_reduction_per_time_unit_per_ev = -0.15
charging_soc_mw_addition_to_demand_per_time_unit_per_ev = 0.01
discharging_soc_mw_reduction_from_demand_per_time_unit_per_ev = 0.01
driving_soc_reduction_per_km_per_ev = 0.0035
forecast_flag = False
n_percent_honesty = ['0.25', '0.5', '0.75']

# Time conversion
index_of_time = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
time_of_day = [17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]

index_to_time_of_day_dict = {}
for item in range(len(index_of_time)):
    index_to_time_of_day_dict[index_of_time[item]] = time_of_day[item]
pprint(index_to_time_of_day_dict)

# Define experiment params
experiment_params = {'n_episodes': n_episodes,
                     'n_hours': 15,
                     'n_divisions_for_soc': 4,
                     'n_divisions_for_percent_honesty': 3,
                     'max_soc_allowed': 1,
                     'min_soc_allowed': 0.1,
                     'alpha': 0.01,
                     'epsilon': 0.1,
                     'gamma': 1,
                     'total_cars_in_alberta': 1000000 / scale,
                     'ev_market_penetration': pen,
                     'charging_soc_addition_per_time_unit_per_ev': 0,
                     'discharging_soc_reduction_per_time_unit_per_ev': 0,
                     'charging_soc_mw_addition_to_demand_per_time_unit_per_ev': 0,
                     'discharging_soc_mw_reduction_from_demand_per_time_unit_per_ev': 0,
                     'driving_soc_reduction_per_km_per_ev': driving_soc_reduction_per_km_per_ev,
                     'alberta_average_demand': alberta_avg_demand,
                     'index_to_time_of_day_dict': index_to_time_of_day_dict,
                     'forecast_flag': forecast_flag,
                     'n_percent_honesty': n_percent_honesty,
                     'which_avg_param': avg
                     }

# Experiment function
class Experiment():

    def __init__(self, experiment_params={}):

        # Initialize all experiment params
        self.n_episodes = experiment_params.get('n_episodes')
        self.n_hours = experiment_params.get('n_hours')
        self.n_divisions_for_soc = experiment_params.get('n_divisions_for_soc')
        self.n_divisions_for_percent_honesty = experiment_params.get('n_divisions_for_percent_honesty')
        self.max_soc_allowed = experiment_params.get('max_soc_allowed')
        self.min_soc_allowed = experiment_params.get('min_soc_allowed')
        self.alpha = experiment_params.get('alpha')
        self.epsilon = experiment_params.get('epsilon')
        self.gamma = experiment_params.get('gamma')
        self.total_cars_in_alberta = experiment_params.get('total_cars_in_alberta')
        self.ev_market_penetration = experiment_params.get('ev_market_penetration')
        self.charging_soc_addition_per_time_unit_per_ev = experiment_params.get('charging_soc_addition_per_time_unit_per_ev')
        self.discharging_soc_reduction_per_time_unit_per_ev = experiment_params.get('discharging_soc_reduction_per_time_unit_per_ev')
        self.charging_soc_mw_addition_to_demand_per_time_unit_per_ev = experiment_params.get('charging_soc_mw_addition_to_demand_per_time_unit_per_ev')
        self.discharging_soc_mw_reduction_from_demand_per_time_unit_per_ev = experiment_params.get('discharging_soc_mw_reduction_from_demand_per_time_unit_per_ev')
        self.driving_soc_reduction_per_km_per_ev = experiment_params.get('driving_soc_reduction_per_km_per_ev')
        self.alberta_average_demand = experiment_params.get('alberta_average_demand')
        self.index_to_time_of_day_dict = experiment_params.get('index_to_time_of_day_dict')
        self.forecast_flag = experiment_params.get('forecast_flag')
        self.n_percent_honesty = experiment_params.get('n_percent_honesty')
        self.which_avg_param = experiment_params.get('which_avg_param')

        # Initialize Q values
        self.Q = np.zeros((self.n_divisions_for_soc, self.n_divisions_for_percent_honesty, self.n_hours, 2))

    def start_experiment(self):

        # Initialize stats
        self.stats = {
            'rewards': [],
            'PARs': [],
            'mean_SOC': [],
            'final_SOC': [],
            'actions': []
        }

    def run(self):

        self.start_experiment()

        for ep in tqdm(range(self.n_episodes)):

            # Initialize state
            soc = np.random.uniform(self.min_soc_allowed, self.max_soc_allowed)
            percent_honesty = np.random.randint(0, self.n_divisions_for_percent_honesty)
            current_hour = np.random.randint(0, self.n_hours)
            done = False

            total_reward = 0

            while not done:

                # Choose action
                if np.random.uniform() < self.epsilon:
                    action = np.random.randint(0, 2)
                else:
                    action = np.argmax(self.Q[int(soc * self.n_divisions_for_soc), percent_honesty, current_hour])

                # Calculate next state
                next_soc = soc + self.charging_soc_addition_per_time_unit_per_ev * action \
                            + self.discharging_soc_reduction_per_time_unit_per_ev * (1 - action)

                next_soc = max(self.min_soc_allowed, min(self.max_soc_allowed, next_soc))

                next_hour = (current_hour + 1) % self.n_hours

                # Calculate reward
                reward = self.alberta_average_demand[current_hour] * (soc - next_soc) \
                         - abs(self.alberta_average_demand[current_hour] * (soc - next_soc)) * (1 - action)

                # Update Q-values
                self.Q[int(soc * self.n_divisions_for_soc), percent_honesty, current_hour, action] += self.alpha * (
                            reward + self.gamma * np.max(
                        self.Q[int(next_soc * self.n_divisions_for_soc), percent_honesty, next_hour]) -
                            self.Q[int(soc * self.n_divisions_for_soc), percent_honesty, current_hour, action])

                total_reward += reward

                soc = next_soc
                current_hour = next_hour

                if current_hour == 0:
                    done = True

            # Record stats
            self.stats['rewards'].append(total_reward)
            PAR = total_reward / (self.alberta_average_demand.mean() * (1 - self.ev_market_penetration))
            self.stats['PARs'].append(PAR)
            self.stats['mean_SOC'].append(soc)
            self.stats['final_SOC'].append(soc)
            self.stats['actions'].append(action)

        print("Experiment done.")

        # Save and print stats
        stats_df = pd.DataFrame(self.stats)
        stats_df.to_csv(f"stats_output_without_v2g_{id_run}.csv", index=False)
        print("Stats saved to stats_output_without_v2g.csv")
        pprint(stats_df.describe())


if __name__ == "__main__":
    experiment = Experiment(experiment_params)
    experiment.run()
