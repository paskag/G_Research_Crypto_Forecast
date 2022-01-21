import pandas as pd
from sklearn.preprocessing import StandardScaler


def load(train_path, details_path):
    df = pd.read_csv(train_path, nrows=100000)
    asset_details = pd.read_csv(details_path)
    return df, asset_details


def preprocess(df, asset_details):
    asset_details = asset_details.set_index("Asset_ID")
    #добавляем название крипты
    df["Name"] = df["Asset_ID"].apply(lambda x: asset_details.loc[x][-1])
    df = df.set_index("Name").reset_index()
    #удаление строк с недостающими target
    df_without_null = df.drop(df[df["Target"].isnull()]["Name"].keys(), axis=0)
    return df_without_null


def main():
    df, asset_details = load("train.csv", "asset_details.csv")
    df_without_null = preprocess(df, asset_details)
    print(df_without_null.head(5))


if __name__ == "__main__":
    main()