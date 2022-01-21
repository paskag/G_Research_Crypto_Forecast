import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from preprocess import load, preprocess

def ml_crypto(name, df_without_null):
    crypto = df_without_null[df_without_null["Name"] == name]
    features = df_without_null.drop(["timestamp", "Asset_ID", "Count", "Name", "Target"], axis=1).columns
    sc = StandardScaler()
    crypto_scaled = sc.fit_transform(crypto.drop(["timestamp", "Asset_ID", "Count", "Name", "Target"], axis=1))
    crypto_scaled = pd.DataFrame(crypto_scaled, columns=features)
    crypto_scaled["Target"] = list(crypto["Target"])
    X = crypto_scaled.drop("Target", axis=1)
    y = crypto["Target"]
    X_train, y_train = X[:int(X.shape[0] * 0.8)], y[:int(X.shape[0] * 0.8)]
    X_valid, y_valid = X[int(X.shape[0] * 0.8):], y[int(X.shape[0] * 0.8):]
    #линейная регрессия
    """lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_mae = round(mean_absolute_error(y_valid, lr.predict(X_valid)), 7)"""
    #случайный лес
    rf = RandomForestRegressor(n_estimators=50, max_depth=3, max_features='log2', criterion="absolute_error")
    rf.fit(X_train, y_train)
    return rf


def main():
    df, asset_details = load("train.csv", "asset_details.csv")
    df_without_null = preprocess(df, asset_details)
    crypto = ml_crypto("Bitcoin", df_without_null)
    print(crypto)


if __name__ == "__main__":
    main()

