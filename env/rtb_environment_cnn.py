import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from gym.envs.registration import EnvSpec
import math

MAX_DAILY_BUDGET = 20000  # maximum daily budget
MAX_STEPS = 12 * 24  # i.e. 288 5 min intervals
MAX_REWARD = 1.0
MAX_BID_THRESH = 0.02
STATE_LOOKBACK = 6
BUDGET_MULT = 1.25

class HFTEnvironment:
    pass


class RTBEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}
    spec = EnvSpec("RTBEnv-v0")

    def __init__(self, df, external_model=False, reset_budget=True):
        self.df = df
        self.reward_range = (0, MAX_REWARD)
        self.action_space = spaces.Box(low=0.0, high=MAX_BID_THRESH, shape=(1,), dtype=np.float)
        # self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7, STATE_LOOKBACK), dtype=np.float) #box range [0,1] due to usage of min-max saclar
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(STATE_LOOKBACK, 7), dtype=np.float)
        self.external_model = external_model

    def _encode_state(self, mode="not_flatten"):

        obs = np.array([self.episode_df.tc_prev.iloc[self.current_step - STATE_LOOKBACK:self.current_step].values,
                        self.episode_df.ctr_prev.iloc[self.current_step - STATE_LOOKBACK:self.current_step].values,
                        self.episode_df.ctr_pred.iloc[self.current_step - STATE_LOOKBACK:self.current_step].values,
                        self.episode_df.total_bids_prev.iloc[
                        self.current_step - STATE_LOOKBACK:self.current_step].values,
                        self.episode_df.tod.iloc[self.current_step - STATE_LOOKBACK:self.current_step].values,
                        self.episode_df.avg_bid_price_prev.iloc[
                        self.current_step - STATE_LOOKBACK:self.current_step].values])
        obs = np.c_[obs, np.asarray([self.budget] * STATE_LOOKBACK)]
        if mode != "not_flatten":
            obs = obs.flatten()
        # obs = obs.reshape(1, 6, 7)
        return obs

    def _gen_action(self, action):
        """
        if agent bid price is greater than average bid price,
        then assign the bid adjusted reward to the agent
        Carefully analyze boundry conditions, i.e. if bid>available budget, then bid=cur budeget
        :param action:
        :return:
        """
        if action > self.budget:
            action = self.budget
        elif action < 0:
            action = 0

        return action

    def step(self, action):
        """
        most critical aspect of the environment, carefully decide how you want to assign the
        reward to the agent based on the current action. The reward will be assigned only if
        bid price is greater than average bid price for the current 5 min period
        :param action:
        :return:
        """
        # print(f'steps {self.current_step}, {self.episode_df.shape[0]}')
        if self.external_model:
            action = self._gen_action(action)
        else:
            action = self._gen_action(action * MAX_BID_THRESH * self.max_budget)

        if action < self.episode_df.cur_avg_bid_price.iloc[self.current_step]:
            reward = 0
        else:
            reward = min(1, self.episode_df.ctr.iloc[self.current_step] / action)
            self.budget -= action

        # add terminal reward penalty if agent is timid

        if self.budget is np.NaN or self.budget is np.Inf:
            print(f'inf nan budget is {self.budget}')

        done = self.budget <= 0 or self.current_step >= (self.episode_df.shape[0] - 3)

        # #adding a terminal penalty if budget is not exhausted
        # if done and self.budget>0:
        #     reward-=(self.budget/self.max_budget)**2

        self.current_step += 1
        obs = self._encode_state()
        # obs = obs.flatten()

        info = {
            "timestamp": self.episode_df.datetime.iloc[self.current_step - 1],
            "remaining budget": self.budget
        }

        return obs, reward, done, info

    def reset(self):
        self.episode_day = np.random.choice(self.df.episode_key.values.tolist())
        self.episode_df = self.df.loc[self.df["episode_key"] == self.episode_day]
        self.episode_df.index = list(range(self.episode_df.shape[0]))
        # self.budget = MAX_DAILY_BUDGET # change budget here
        self.max_budget = self.budget = self.episode_df["daily_budget"].iloc[0]*BUDGET_MULT
        self.total_reward = 0
        self.current_step = STATE_LOOKBACK  # add a random start point
        return self._encode_state()

    @classmethod
    def load_from_h5(cls, h5_path, table_name, external_model=False):
        with pd.HDFStore(h5_path, mode='r') as store:
            df = store.select(table_name)
        return RTBEnvironment(df, external_model)


if __name__ == '__main__':
    store_loc = "../data/rtb_store_full_5min.h5"
    env = RTBEnvironment.load_from_h5(store_loc, "full_df")

    # todo: run the environment with random actions
    pass





