from env import rtb_hft_environment as rtb_env
import pandas as pd
import numpy as np
np.random.seed(7)

STORE_LOCATION = "../data/rtb_1458_store.h5"
HFT_STORE_LOCATION = "../data/rtb_1458_hft_store.h5"
TABLE_NAME = "df_1458"
EPISODES = 10
RESULTS_PATH =  "../results/random_action/"


def random_actions(remaining_budget):
    max_5min_budget = rtb_env.MAX_DAILY_BUDGET*rtb_env.MAX_BID_THRESH
    random_value = np.random.rand()
    random_action = min((random_value*(max_5min_budget)), remaining_budget)
    return random_action


def run_random_agent():
    env = rtb_env.RTBEnvironment.load_from_h5(STORE_LOCATION, HFT_STORE_LOCATION, TABLE_NAME)
    reward_dict = {}
    for i in range(EPISODES):
        total_episode_reward = 0
        done = False
        obs = env.reset()

        while not done:
            action = random_actions(env.budget)
            obs, reward, done, info = env.step(action)
            total_episode_reward+=reward
            print(f"Episode: {i}, TimeStamp: {info['timestamp']}, Reward:{reward}, remaining budget (%): "
                  f"{(info['remaining budget']/rtb_env.MAX_DAILY_BUDGET)*100}")

        reward_dict[f"EPISODE_{i}"] = [total_episode_reward]
    pd.DataFrame.from_dict(reward_dict).T.to_csv(f"{RESULTS_PATH}random_5min_hft_agent_w_ctr_model.csv")


if __name__ == '__main__':
    run_random_agent()