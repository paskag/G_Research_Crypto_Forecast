from calendar import c
from statistics import mode
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from preprocess import load, preprocess
from train import ml_crypto


def save_models():
    df, asset_details = load("train.csv", "asset_details.csv")
    df_without_null = preprocess(df, asset_details)
    crypto_names = set(df_without_null["Name"])
    models = {}
    for crypto_name in crypto_names:
        models[crypto_name] = ml_crypto(crypto_name, df_without_null)


def main():
    save_models()


if __name__ == "__main__":
    main()
