import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from predict import prediction
from sklearn.preprocessing import PolynomialFeatures


def visualization(name, df_without_null_scaled, method):
    model = load(f"saved_models/{name}_{method}")
    crypto = df_without_null_scaled[df_without_null_scaled["Name"] == name]
    crypto = crypto.drop("Name", axis=1)
    x = crypto.drop(["timestamp", "Target"], axis=1)
    y = crypto["Target"]
    X_valid, y_valid = x[int(x.shape[0] * 0.8):], y[int(x.shape[0] * 0.8):]
    if method == "svr" or "SVR":
        y_valid = y_valid * 1000
    if method == "polynom":
        pol = PolynomialFeatures(degree=2)
        X_valid = pol.fit_transform(X_valid)
    pred, model_mae = prediction(model, X_valid, y_valid)
    
    print(f"Для криптовалюты {name} MAE на {method}: {model_mae}")
    plt.figure(figsize=(15, 7))
    sns.lineplot(x=crypto["timestamp"][int(x.shape[0] * 0.8):], y=y_valid)
    sns.lineplot(x=crypto["timestamp"][int(x.shape[0] * 0.8):], y=pred)
    plt.title(f"График предсказания стоимости криптовалыты: {name}")
    plt.ylabel("Стоимость")
    plt.legend(["Actual", f"Predicted {method}"])
