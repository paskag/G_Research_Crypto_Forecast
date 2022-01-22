import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from preprocess import load, preprocess
from config import config_model

def train_crypto(name, df):
    crypto = df[df["Name"] == name]
    crypto = crypto.drop("Name", axis=1)
    X = crypto.drop("Target", axis=1)
    y = crypto["Target"]
    X_train, y_train = X[:int(X.shape[0] * 0.8)], y[:int(X.shape[0] * 0.8)]
    X_valid, y_valid = X[int(X.shape[0] * 0.8):], y[int(X.shape[0] * 0.8):]
    #линейная регрессия
    #lr = LinearRegression()
    #lr.fit(X_train, y_train)
    #lr_mae = round(mean_absolute_error(y_valid, lr.predict(X_valid)), 7)
    #случайный лес
    #rf = RandomForestRegressor(n_estimators=50, max_depth=3, max_features='log2', criterion="absolute_error")
    model = RandomForestRegressor(**config_model(name))
    model.fit(X_train, y_train)
    return model


def main():
    df, asset_details = load("train.csv", "asset_details.csv")
    df_without_null_scaled = preprocess(df, asset_details)
    crypto = train_crypto("Bitcoin", df_without_null_scaled)
    print(crypto)


if __name__ == "__main__":
    main()

