from test_agents.test_environment_lstm import RTBEnvironment as rtb_env
from test_agents.test_environment import MAX_DAILY_BUDGET, MAX_BID_THRESH
import pandas as pd
import numpy as np
import torch
from models import model, utils
import random
import ptan

# MODEL_PATH = "/Users/sujitkhanna/Desktop/Courses/capstone/code base/rtb_exploration/agents/saves/a2c-/model_DeepModel multiple lbk unique budget tanh action -a2c_l_r: 0.0005, en_b :0.1, max_ep:4000, max_it: 100, reward_steps: 48  batch_size: 16best_+0.299_363900.dat"
# MODEL_PATH = "/Users/sujitkhanna/Desktop/Courses/capstone/code base/rtb_exploration/agents/saves/a2c-/best_+0.298_70800.dat"
MODEL_PATH = "/Users/sujitkhanna/Desktop/Courses/capstone/code base/rtb_exploration/agents/saves/a2c-/model_ LSTM RNN unique budget tanh action -a2c_l_r: 0.0001, en_b :0.01, max_ep: 3000, max_it: 100, reward_steps: 12  batch_size: 16best_+0.053_131100.dat"
STORE_LOCATION = "../data/rtb_store_full_test_5min.h5"
TABLE_NAME = "full_df_w_unique_key"
EPISODES = 15
RESULTS_PATH =  "../results/random_action/"


def run_mlp_agent():
    device = torch.device("cpu")
    env = rtb_env.load_from_h5(STORE_LOCATION, TABLE_NAME)
    lstm_units = 10
    net = model.LSTMModelA2C(env.observation_space.shape[1], env.action_space.shape[0], lstm_units).to(device)
    net.load_state_dict(torch.load(MODEL_PATH))
    reward_list, budget_list, episode_list = [], [], []
    for i in range(EPISODES):
        total_episode_reward = 0
        done = False
        obs = env.reset()
        while not done:
            # obs_v = torch.FloatTensor([obs])
            obs_v = ptan.agent.float32_preprocessor([obs])
            mu_v, var_v, val_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            obs, reward, done, info = env.step(action[0])
            total_episode_reward += reward
            print(f"Episode: {i}, TimeStamp: {info['timestamp']}, Reward:{reward}, remaining budget (%): "
                  f"{(info['remaining budget']) * 100}")

        reward_list.append(total_episode_reward)
        budget_list.append((info['remaining budget']))
        episode_list.append(f"EPISODE_{i}")
    print(f'average reward for test set is {sum(reward_list) / len(reward_list)}')
    pd.DataFrame({"Episodes": episode_list, "rewards": reward_list, "rem_budget":budget_list}).to_csv(f"{RESULTS_PATH}lstm_rnn_5min_agent_testset.csv")


if __name__ == '__main__':
    run_mlp_agent()