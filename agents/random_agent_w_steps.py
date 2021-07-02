from env import rtb_environment as rtb_env
import pandas as pd
import numpy as np

import os
import time
import math
import ptan
import gym
import argparse
from tensorboardX import SummaryWriter

from models import model, utils

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

#tensorboard --logdir=runs/ --host localhost --port 8088

SAVEPATH = ""
# STORE_LOCATION = "../data/rtb_1458_store.h5"
# TABLE_NAME = "df_1458"
STORE_LOCATION = "../data/rtb_store_full_train_5min.h5"
TABLE_NAME = "full_df_w_unique_key"
MAX_EPISODES=3000

ENV_ID = "RTBEnv-v0"
GAMMA = 0.99
REWARD_STEPS = 48 #how many steps ahead bellman's equation is unrolled to estimate discounted total reward of every transaction
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
ENTROPY_BETA = 5e-3

TEST_ITERS = 10000

def random_actions(remaining_budget):
    max_5min_budget = rtb_env.MAX_DAILY_BUDGET*rtb_env.MAX_BID_THRESH
    random_value = np.random.rand()
    random_action = min((random_value*(max_5min_budget)), remaining_budget)
    return random_action


def test_net(net, env, count=10, device="cpu"):
    """
     perform periodical tests of our model on the separate testing environment.
     During the testing, we don't need to do any exploration; we will just use
     the mean value returned by the model directly, without any random sampling.
     I.e this is the target policy and not the behavior policy
    :param net:
    :param env:
    :param count:
    :param device:
    :return:
    """
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs])
            obs_v = obs_v.to(device)
            if len(obs_v.shape) > 2:
                obs_v = obs_v.resize(1, 7)

            import random
            action = random.uniform(0, 1)

            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def calc_logprob(mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2


if __name__ == "__main__":
    device = torch.device("cpu")
    save_path = os.path.join("saves", "a2c-" + SAVEPATH)
    os.makedirs(save_path, exist_ok=True)

    env = rtb_env.RTBEnvironment.load_from_h5(STORE_LOCATION, TABLE_NAME)
    test_env = rtb_env.RTBEnvironment.load_from_h5(STORE_LOCATION, TABLE_NAME)

    net = model.ModelA2C(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(net)

    writer = SummaryWriter(comment="-random_w-unique budget" + SAVEPATH)

    agent = model.AgentA2C(net, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    batch = []
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                rewards_steps = exp_source.pop_rewards_steps() #once the experience replay is full
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], step_idx)
                    tracker.reward(rewards[0], step_idx)

                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_net(net, test_env, device=device)
                    print("Test done is %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(net.state_dict(), fname)
                        best_reward = rewards

                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = \
                    utils.unpack_batch_a2c(
                        batch, net, device=device,
                        last_val_gamma=GAMMA ** REWARD_STEPS)
                batch.clear()

                optimizer.zero_grad()
                mu_v, var_v, value_v = net(states_v)

                loss_value_v = F.mse_loss(
                    value_v.squeeze(-1), vals_ref_v) #critic loss

                adv_v = vals_ref_v.unsqueeze(dim=-1) - \
                        value_v.detach()
                log_prob_v = adv_v * calc_logprob(
                    mu_v, var_v, actions_v)
                loss_policy_v = -log_prob_v.mean() # this is the policy loss
                ent_v = -(torch.log(2*math.pi*var_v) + 1)/2
                entropy_loss_v = ENTROPY_BETA * ent_v.mean() # this is the entropy loss

                loss_v = loss_policy_v + entropy_loss_v + \
                         loss_value_v #overall loss
                loss_v.backward()
                optimizer.step()

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
                tb_tracker.track("mean_action", mu_v, step_idx)
                tb_tracker.track("variance_action", var_v, step_idx)
                tb_tracker.track("action_value", actions_v, step_idx)

                if len(tracker.total_rewards)==MAX_EPISODES:
                    break

        print(f"best reward after {MAX_EPISODES} episodes is: {best_reward}, max_reward = {max(tracker.total_rewards)}")
