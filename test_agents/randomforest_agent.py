from test_agents.test_environment import RTBEnvironment as rtb_env
from test_agents.test_environment import MAX_DAILY_BUDGET, MAX_BID_THRESH
import pandas as pd
import numpy as np
import torch
from models import model, utils
import random
import ptan
import joblib


STORE_LOCATION = "../data/rtb_store_full_test_5min.h5"
TABLE_NAME = "full_df_w_unique_key"
TRAINED_MODEL = "../saved_models/RF_full_bid_price_regression_trunc.pkl"
EPISODES = 15
RESULTS_PATH =  "../results/random_action/"
MODEL = joblib.load(TRAINED_MODEL)
def randomforest_action(obs):
    obs = obs.astype(float)
    obs = obs[:-1]
    obs = obs.reshape(1, -1)
    action = MODEL.predict(obs)
    return action[0]


def run_random_agent():
    env = rtb_env.load_from_h5(STORE_LOCATION, TABLE_NAME, external_model=True)
    reward_list,budget_list,episode_list = [], [], []
    step_reward, step_action = [], []
    for i in range(EPISODES):
        total_episode_reward = 0
        done = False
        obs = env.reset()

        while not done:
            action = randomforest_action(obs)
            obs, reward, done, info, act = env.step(action)
            total_episode_reward+=reward
            print(f"Episode: {i}, TimeStamp: {info['timestamp']}, Reward:{reward}, remaining budget (%): "
                  f"{(info['remaining budget'])*100}")
            step_action.append(act)
            step_reward.append(reward)

        reward_list.append(total_episode_reward)
        budget_list.append((info['remaining budget']))
        episode_list.append(f"EPISODE_{i}")
        print(f'average reward for test set is {sum(reward_list) / len(reward_list)}')
        pd.DataFrame({"Episodes": episode_list, "rewards": reward_list, "rem_budget": budget_list}).to_csv(
            f"{RESULTS_PATH}random_forest_0.75x_budget_agent_testset.csv")

        pd.DataFrame({"step_rewards": step_reward, "step_actions": step_action}).to_csv(
            f"{RESULTS_PATH}random_forest_0.75x_budget_agent_testset_step_data.csv")


if __name__ == '__main__':
    run_random_agent()
