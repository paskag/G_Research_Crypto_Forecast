import pandas as pd
from sklearn.preprocessing import StandardScaler


def load(train_path, details_path):
    df = pd.read_csv(train_path, nrows=100000)
    asset_details = pd.read_csv(details_path)
    return df, asset_details


def preprocess(df, asset_details):
    asset_details = asset_details.set_index("Asset_ID")
    #добавляем название крипты
    df.insert(0, "Name", df["Asset_ID"].apply(lambda x: asset_details.loc[x][-1]))
    #удаление строк с недостающими target
    df_without_null = df.drop(df[df["Target"].isnull()]["Name"].keys(), axis=0)
    #StandardScaler
    features = df_without_null.drop(["timestamp", "Asset_ID", "Count", "Name", "Target"], axis=1).columns
    sc = StandardScaler()
    df_without_null_scaled = sc.fit_transform(df_without_null.drop(["timestamp", "Asset_ID", "Count", "Name", "Target"], axis=1))
    df_without_null_scaled = pd.DataFrame(df_without_null_scaled, columns=features)
    df_without_null_scaled["Target"] = list(df_without_null["Target"])
    df_without_null_scaled.insert(0, "Name", list(df_without_null["Name"]))
    return df_without_null_scaled


def main():
    df, asset_details = load("train.csv", "asset_details.csv")
    df_without_null_scaled = preprocess(df, asset_details)
    print(df_without_null_scaled.head(10))


if __name__ == "__main__":
    main()
