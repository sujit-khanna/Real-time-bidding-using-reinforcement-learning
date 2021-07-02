import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from gym.envs.registration import EnvSpec
import catboost as cb
from catboost import CatBoostClassifier

MAX_DAILY_BUDGET = 20000 # maximum daily budget
MAX_STEPS = 12*24 # i.e. 288 5 min intervals
MAX_REWARD = 1.0
MAX_BID_THRESH = 0.03
STATE_LOOKBACK = 1
HFT_CTR_MODEL_PATH = "../saved_models/hft_ctr_catboost"

class HFTEnvironment:
    pass


class RTBEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}
    spec = EnvSpec("RTBHFTEnv-v0")

    def __init__(self, df, df_hft, reset_budget=True):
        self.df = df
        self.df_hft = df_hft.reset_index()
        self.reward_range = (0, MAX_REWARD)
        self.action_space = spaces.Box(low=0.0, high=MAX_BID_THRESH, shape=(1,), dtype=np.float)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7, STATE_LOOKBACK), dtype=np.float) #box range [0,1] due to usage of min-max saclar

    def _encode_state(self):
        obs = np.array([self.episode_df.tc_prev.iloc[self.current_step-STATE_LOOKBACK:self.current_step].values,
                                  self.episode_df.ctr_prev.iloc[self.current_step-STATE_LOOKBACK:self.current_step].values,
                                  self.episode_df.ctr_pred.iloc[self.current_step-STATE_LOOKBACK:self.current_step].values,
                                  self.episode_df.total_bids_prev.iloc[self.current_step - STATE_LOOKBACK:self.current_step].values,
                                  self.episode_df.tod.iloc[self.current_step-STATE_LOOKBACK:self.current_step].values,
                                  self.episode_df.avg_bid_price_prev.iloc[self.current_step-STATE_LOOKBACK:self.current_step].values,
                                  self.budget])
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
        elif action<0:
            action = 0

        return action

    def _process_hft_data(self, df):
        ip_features = ["weekday", "hour", "region", "slotwidth", "slotheight", "slotvisibility", "slotformat",
                       "slotprice", "useragent"]
        op_feature = ["click"]
        df["useragent"] = pd.factorize(df['useragent'])[0] + 1
        df["slotarea"] = df["slotwidth"] * df["slotheight"]
        ip_features.remove("slotwidth")
        ip_features.remove("slotheight")
        df_short = df[ip_features + ["slotarea"] + op_feature]
        X, y = df_short[ip_features], df_short[op_feature]
        from_file = CatBoostClassifier()
        from_file.load_model(HFT_CTR_MODEL_PATH)
        test_preds_prob = from_file.predict_proba(X)
        df["ctr_probs"] = test_preds_prob[:, 1]

        return df

    def _hft_step(self, hft_df, action, prob_tresh=0.95):
        reward, bid_id = 0, 0
        bid_flag = False
        hft_df = self._process_hft_data(hft_df)
        for i in range(hft_df.shape[0]):
            if hft_df["ctr_probs"].iloc[i]>=prob_tresh:
                if action>=hft_df["payprice"].iloc[i]:
                    click = hft_df["click"].iloc[i]
                    reward = (click/action)*100
                    bid_flag = True
                    bid_id = hft_df["bidid"].iloc[i]
                    break

        return reward, bid_flag, bid_id

    def step(self, action):
        """
        For HFT environment; once a five minute sub-set is chosen,
        1. slice the start and end periods from the HFT dataframe
        2. enter the HFT and start bidding; if any price matches i.e price>=bids
        3. if bid matches: assign the reward; else
        4. move to next hft time step
        :param action:
        :return:
        """

        action = self._gen_action(action)

        hft_df = self.df_hft.loc[(self.df_hft["datetime"]<=self.episode_df["datetime"].iloc[self.current_step] + pd.Timedelta(minutes=5))
                                 & (self.df_hft["datetime"]>=self.episode_df["datetime"].iloc[self.current_step])]

        hft_df.index = list(range(hft_df.shape[0]))

        reward, bid_flag, bid_id = self._hft_step(hft_df, action)
        if bid_flag: self.budget-=action

        done = self.budget<=0 or self.current_step>=self.episode_df.shape[0]

        self.current_step+=1
        obs = self._encode_state()

        info = {
            "timestamp": self.episode_df.datetime.iloc[self.current_step],
            "remaining budget": self.budget,
            "bid_id":bid_id
        }

        return obs, reward, done, info

    def reset(self):
        self.episode_day = np.random.choice(self.df.episode_key.values.tolist())
        self.episode_df = self.df.loc[self.df["episode_key"]==self.episode_day]
        self.episode_df.index = list(range(self.episode_df.shape[0]))
        self.budget = MAX_DAILY_BUDGET
        self.total_reward = 0
        self.current_step = 1 #add a random start point
        return self._encode_state()

    @classmethod
    def load_from_h5(cls, h5_path, hft_h5_path, table_name):
        with pd.HDFStore(h5_path, mode='r') as store:
            df = store.select(table_name)
            store.close()
        with pd.HDFStore(hft_h5_path, mode='r') as store:
            df_hft = store.select(table_name)
            store.close()
        return RTBEnvironment(df, df_hft)


if __name__ == '__main__':
    store_loc = "../data/rtb_store.h5"
    store_loc = "../data/rtb_store.h5"
    env = RTBEnvironment.load_from_h5(store_loc, "df_1458", "hft_df_1458")

    #todo: run the environment with random actions
    pass





