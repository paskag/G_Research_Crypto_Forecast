from joblib import dump


def save_model(name, model):
    dump(model, f"saved_models/{name}_model")
