import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ydata_profiling

# Pandas Profiling
train = pd.read_csv("../input/titanic/train.csv")
ydata_profiling.ProfileReport(train).to_file("pandas_profile.html")

# AgeとSurvivedの関係
plt.hist(
    train.loc[train["Survived"] == 0, "Age"].dropna(), bins=30, alpha=0.5, label="0"
)
plt.hist(
    train.loc[train["Survived"] == 1, "Age"].dropna(), bins=30, alpha=0.5, label="1"
)
plt.xlabel("Age")
plt.ylabel("count")
plt.legend(title="Survived")
plt.savefig("graph_age.png")

# SibSp(兄弟姉妹と配偶者の人数)とSurvivedの関係
sns.countplot(x="SibSp", hue="Survived", data=train)
plt.legend(loc="upper right", title="Survived")
plt.savefig("graph_SibSp.png")

# Parch(同乗した親と子供の人数)とSurvivedの関係
sns.countplot(x="Parch", hue="Survived", data=train)
plt.legend(loc="upper right", title="Survived")
plt.savefig("graph_parch.png")

# FareとSurvivedの関係
plt.hist(
    train.loc[train["Survived"] == 0, "Fare"].dropna(),
    range=(0, 250),
    bins=25,
    alpha=0.5,
    label="0",
)
plt.hist(
    train.loc[train["Survived"] == 1, "Fare"].dropna(),
    range=(0, 250),
    bins=25,
    alpha=0.5,
    label="1",
)
plt.xlabel("Fare")
plt.ylabel("count")
plt.legend(title="Survived")
plt.xlim(-5, 250)
plt.savefig("graph_fare.png")

# Pclass(チケットのクラス)とSurvivedの関係
sns.countplot(x='Pclass', hue='Survived', data=train)
plt.savefig("graph_pclass.png")

# SexとSurvivedの関係
sns.countplot(x='Sex', hue='Survived', data=train)
plt.savefig("graph_sex.png")

# Embarked(乗船した港)とSurvivedの関係
sns.countplot(x='Embarked', hue='Survived', data=train)
plt.savefig("graph_embarked.png")

# FamilySizeを作成してSurvivedとの関係を見る
train['FamilySize'] = train['Parch'] + train['SibSp'] + 1
sns.countplot(x='FamilySize', hue='Survived', data=train)
plt.savefig("graph_FamilySize.png")

# IsAloneを作成してSurvivedとの関係を見る
train['IsAlone'] = 0
train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
sns.countplot(x='IsAlone', hue='Survived', data=train)
plt.savefig("graph_IsAlone.png")
