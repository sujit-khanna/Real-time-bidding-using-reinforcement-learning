
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

STORE_LOCATION = "../data/rtb_store_full_5min.h5"
TABLE_NAME = "full_df_w_unique_key"


def load_data():
    with pd.HDFStore(STORE_LOCATION, mode='r') as store:
        df = store.select(TABLE_NAME)

    return df


def train_regression_model(df, feature_list, target_list,model_type="RF"):
    X, y = df[feature_list], df[target_list]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.3, test_size=0.7, random_state=4)
    if model_type=="RF":
        model = RandomForestRegressor(max_depth=2, random_state=0)
        model.fit(X_train.values, y_train.values)
        joblib.dump(model, "../saved_models/RF_bid_price_regression_trunc.pkl")
    elif model_type=="catboost":
        model = CatBoostRegressor(iterations=10000,
                                  learning_rate=0.02)
        model.fit(X_train.values, y_train.values, verbose=False)
        save_model_path = "../saved_models/CB_bid_price_regression_trunc2"
        model.save_model(save_model_path)


if __name__ == '__main__':
    feature_list = ["tc_prev", "ctr_prev", "ctr_pred", "total_bids_prev",
                      "tod", "avg_bid_price_prev"]
    target_value = "cur_avg_bid_price"
    df = load_data()
    train_regression_model(df, feature_list, target_value, model_type="catboost")
