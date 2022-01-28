import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from preprocess import load, preprocess
from config import config_rf, config_svr
from save import save_model
import joblib

def train_rf(name, X_train, y_train):
    #случайный лес
    model = RandomForestRegressor(**config_rf(name), n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_lr(X_train, y_train):
    #линейная регрессия
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_polynom(X_train, y_train):
    #полином 2 степени
    pol = PolynomialFeatures(degree=2)
    X_train_pol = pol.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train_pol, y_train)
    return model


def train_svr(name, X_train, y_train):
    y_train = y_train * 1000
    model = SVR(**config_svr(name))
    model.fit(X_train, y_train)
    return model


def train_test_split(name, df):
    crypto = df[df["Name"] == name]
    crypto = crypto.drop(["timestamp", "Name"], axis=1)
    X = crypto.drop("Target", axis=1)
    y = crypto["Target"]
    X_train, y_train = X[:int(X.shape[0] * 0.8)], y[:int(X.shape[0] * 0.8)]
    X_valid, y_valid = X[int(X.shape[0] * 0.8):], y[int(X.shape[0] * 0.8):]
    return X_train, X_valid, y_train, y_valid


def main():
    #проверка, что все работает
    df, asset_details = load("train.csv", "asset_details.csv")
    df_without_null_scaled = preprocess(df, asset_details)
    X_train, X_valid, y_train, y_valid = train_test_split("Bitcoin", df_without_null_scaled)
    crypto = train_rf("Bitcoin", X_train, y_train)
    save_model("Bitcoin", crypto)
    print(joblib.load("saved_models/Bitcoin_model").predict(X_valid))


if __name__ == "__main__":
    main()
