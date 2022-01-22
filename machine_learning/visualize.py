import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load


def visualization(name, df):
    model = load(f"saved_models/{name}_model")
    crypto = df[df["Name"] == name]
    crypto = crypto.drop("Name", axis=1)
    X = crypto.drop("Target", axis=1)
    y = crypto["Target"]
    X_valid, y_valid = X[int(X.shape[0] * 0.8):], y[int(X.shape[0] * 0.8):]
    plt.figure(figsize=(15, 7))
    sns.lineplot(x=crypto["timestamp"][int(X.shape[0] * 0.8):], y=y_valid)
    #sns.lineplot(x=crypto["timestamp"][int(X.shape[0] * 0.8):], y=lr.predict(X_valid))
    sns.lineplot(x=crypto["timestamp"][int(X.shape[0] * 0.8):], y=model.predict(X_valid))
    plt.title(f"График предсказания стоимости криптовалыты: {name}")
    plt.ylabel("Стоимость")
    plt.legend(["Actual", "Predicted"])
