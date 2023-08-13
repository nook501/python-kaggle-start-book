# https://www.kaggle.com/competitions/titanic/overview
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


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
    data["Age"].fillna(data["Age"].median(), inplace=True)
    data["FamilySize"] = data["Parch"] + data["SibSp"] + 1  # FamilySize を作成
    data["IsAlone"] = 0
    data.loc[data["FamilySize"] == 1, "IsAlone"] = 1  # IsAlone を作成

    delete_columns = ["Name", "PassengerId", "Ticket", "Cabin"]
    data.drop(delete_columns, axis=1, inplace=True)

    train = data[: len(train)]
    test = data[len(train) :]

    return train, test


def objective(trial):
    categorical_features = ["Embarked", "Pclass", "Sex"]
    params = {
        "objective": "binary",
        "max_bin": trial.suggest_int("max_bin", 255, 500),
        "learning_rate": 0.05,
        "num_leaves": trial.suggest_int("num_leaves", 32, 128),
        "random_state": 0,
        "verbose": -1,
    }

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)
    lgb_valid = lgb.Dataset(
        X_valid,
        y_valid,
        reference=lgb_train,
        categorical_feature=categorical_features,
    )

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_valid],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(10),
        ],
    )

    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
    score = log_loss(y_valid, y_pred_valid)  # type: ignore
    return score


# データの読み込み
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")  # Survivedの列だけない
sub = pd.read_csv("../input/titanic/gender_submission.csv")  # 求められる提出の形式のサンプル

# 特徴量エンジニアリング
train, test = feature_engineering(train, test)

# 訓練データ・テストデータの作成
y_train = train["Survived"]
X_train = train.drop("Survived", axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.3, random_state=0, stratify=y_train
)
X_test = test.drop("Survived", axis=1)

# ハイパーパラメータの最適化
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
study.optimize(objective, n_trials=40)  # type: ignore

# モデル作成
categorical_features = ["Embarked", "Pclass", "Sex"]

lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)
lgb_valid = lgb.Dataset(
    X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features
)

params = {"objective": "binary", "random_state": 0, "verbose": -1}
if study.best_params:
    params["max_bin"] = study.best_params["max_bin"]
    params["num_leaves"] = study.best_params["num_leaves"]

model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_valid],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(10),
    ],
)

# 予測
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred > 0.5).astype(int)  # type: ignore

# 提出用ファイルの作成
sub["Survived"] = list(map(int, y_pred))
sub.to_csv("submission_lgb_op.csv", index=False)
