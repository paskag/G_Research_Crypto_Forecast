def config_model(name):
    configs = {
        "Bitcoin": {'max_depth': 3, 'max_features': 'log2', 'n_estimators': 50, "criterion": "absolute_error"},
        "Bitcoin Cash": {'max_depth': 3, 'max_features': 'sqrt', 'n_estimators': 100, "criterion": "absolute_error"},
        "Binance Coin": {'max_depth': 3, 'max_features': 'sqrt', 'n_estimators': 100, "criterion": "absolute_error"},
        "EOS.IO": {'max_depth': 3, 'max_features': 'log2', 'n_estimators': 50, "criterion": "absolute_error"},
        "Ethereum Classic": {'max_depth': 3, 'max_features': 'log2', 'n_estimators': 50, "criterion": "absolute_error"},
        "Ethereum": {'max_depth': 3, 'max_features': 'log2', 'n_estimators': 50, "criterion": "absolute_error"},
        "Litecoin": {'max_depth': 3, 'max_features': 'sqrt', 'n_estimators': 50, "criterion": "absolute_error"},
        "Monero": {'max_depth': 3, 'max_features': 'log2', 'n_estimators': 50, "criterion": "absolute_error"}
    }
    return configs[name]


def main():
    test = config_model("Dogecoin")
    print(test)


if __name__ == "__main__":
    main()