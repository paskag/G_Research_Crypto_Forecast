from joblib import dump


def save_model(name, method, model):
    dump(model, f"saved_models/{name}_{method}")


def main():
    save_model("Bitcoin", "rf", 5)


if __name__ == "__main__":
    main()
