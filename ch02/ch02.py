import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def feature_engineering(train, test):
    """特徴量エンジニアリング"""
    data = pd.concat([train, test], sort=False)
    # print(data.head())
    # print(len(train), len(test), len(data))
    # print(data.isnull().sum())  # 欠損値の数

    data["Sex"].replace(["male", "female"], [0, 1], inplace=True)

    data["Embarked"].fillna(("S"), inplace=True)
    data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)

    data["Fare"].fillna(np.mean(data["Fare"]), inplace=True)  # type: ignore

    age_avg = data["Age"].mean()
    age_std = data["Age"].std()
    data["Age"].fillna(
        np.random.randint(age_avg - age_std, age_avg + age_std),  # type: ignore
        inplace=True,
    )

    delete_columns = ["Name", "PassengerId", "SibSp", "Parch", "Ticket", "Cabin"]
    data.drop(delete_columns, axis=1, inplace=True)

    train = data[: len(train)]
    test = data[len(train) :]

    return train, test


# データの読み込み
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")  # Survivedの列だけない
gender_submission = pd.read_csv(
    "../input/titanic/gender_submission.csv"
)  # 求められる提出の形式のサンプル

# 特徴量エンジニアリング
train, test = feature_engineering(train, test)

# 訓練データ・テストデータの作成
y_train = train["Survived"]
X_train = train.drop("Survived", axis=1)
X_test = test.drop("Survived", axis=1)

# モデル作成
clf = LogisticRegression(penalty="l2", solver="sag", random_state=0, max_iter=10000)
clf.fit(X_train, y_train)

# 予測
y_pred = clf.predict(X_test)
# print(y_pred[:20])

# 提出用ファイルの作成
sub = pd.read_csv("../input/titanic/gender_submission.csv")
sub["Survived"] = list(map(int, y_pred))
sub.to_csv("submission.csv", index=False)
