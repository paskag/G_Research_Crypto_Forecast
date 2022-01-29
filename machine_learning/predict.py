from sklearn.metrics import mean_absolute_error


def prediction(model, X_valid, y_valid):
    prediction = model.predict(X_valid)
    model_mae = round(mean_absolute_error(y_valid, prediction), 7)
    return prediction, model_mae
