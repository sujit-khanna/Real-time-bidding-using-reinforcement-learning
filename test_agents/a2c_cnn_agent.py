from test_agents.test_environment_cnn import RTBEnvironment as rtb_env
import pandas as pd
import torch
from models import model, utils
import ptan


# MODEL_PATH = "/Users/sujitkhanna/Desktop/Courses/capstone/code base/rtb_exploration/agents/saves/a2c-/model_ CNN model unique budget tanh action -a2c_l_r: 0.0001, en_b :0.001, max_ep:2000, max_it: 100, reward_steps: 48  batch_size: 1best_+0.314_147000.dat"
# MODEL_PATH = "/Users/sujitkhanna/Desktop/Courses/capstone/code base/rtb_exploration/agents/saves/a2c-/model_ 0.5x Budget CNN model unique budget tanh action -a2c_l_r: 0.0001, en_b :0.001, max_ep:2000, max_it: 100, reward_steps: 48  batch_size: 1best_+0.280_16600.dat"
# MODEL_PATH = "/Users/sujitkhanna/Desktop/Courses/capstone/code base/rtb_exploration/agents/saves/a2c-/model_ 0.5x Budget CNN model unique budget tanh action -a2c_l_r: 0.0001, en_b :0.01, max_ep:2000, max_it: 100, reward_steps: 36  batch_size: 1best_+0.331_158900.dat"
MODEL_PATH = "/Users/sujitkhanna/Desktop/Courses/capstone/code base/rtb_exploration/agents/saves/a2c-/model_ 1.25x Budget CNN model unique budget tanh action -a2c_l_r: 0.0001, en_b :0.001, max_ep:2500, max_it: 100, reward_steps: 60  batch_size: 1best_+0.282_166700.dat"
STORE_LOCATION = "../data/rtb_store_full_test_5min.h5"
TABLE_NAME = "full_df_w_unique_key"
EPISODES = 15
RESULTS_PATH =  "../results/random_action/"


def run_cnn_agent():
    device = torch.device("cpu")
    env = rtb_env.load_from_h5(STORE_LOCATION, TABLE_NAME)
    net = model.CNNModelA2C(env.observation_space, env.action_space.shape[0]).to(device)
    net.load_state_dict(torch.load(MODEL_PATH))
    reward_list, budget_list, episode_list = [], [], []
    step_reward, step_action = [], []

    for i in range(EPISODES):
        total_episode_reward = 0
        done = False
        obs = env.reset()
        while not done:
            # obs_v = torch.FloatTensor([obs])
            obs_v = ptan.agent.float32_preprocessor([obs])
            mu_v, var_v, val_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            obs, reward, done, info, act = env.step(action[0])
            step_action.append(act)
            step_reward.append(reward)
            total_episode_reward += reward
            print(f"Episode: {i}, TimeStamp: {info['timestamp']}, Reward:{reward}, remaining budget (%): "
                  f"{(info['remaining budget']) * 100}")

        reward_list.append(total_episode_reward)
        budget_list.append((info['remaining budget']))
        episode_list.append(f"EPISODE_{i}")

    print(f'average reward for test set is {sum(reward_list) / len(reward_list)}')

    pd.DataFrame({"Episodes": episode_list, "rewards": reward_list, "rem_budget":budget_list}).to_csv(
        f"{RESULTS_PATH}deep_cnn_1.25x budget_test_5min_agent_testset.csv")

    pd.DataFrame({"step_rewards": step_reward, "step_actions": step_action}).to_csv(
        f"{RESULTS_PATH}deep_cnn_1.25x budget_test_5min_agent_testset_step_data.csv")


if __name__ == '__main__':
    run_cnn_agent()