import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def train():
    # Read Data
    train_df = pd.read_csv('../train.csv')
    test_df = pd.read_csv('../test.csv')

# _________________________________________________DATA CLEANING____________________________________________________
    # Print head and tail of data set:
    # print(train_df.head())
    # print(train_df.tail())

    # See which columns have null
    # train_df.info()
    # print('_'*40)
    # test_df.info()

    # statistic of the data
    # print(train_df.describe())

    # print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False)
    #       .mean().sort_values(by='Survived', ascending=False))

    # graphing correlations
    # g = sns.FacetGrid(train_df, col='Survived', row='Sex', height=2.2, aspect=1.6)
    # g.map(plt.hist, 'Age', alpha=.5, bins=20)
    # g.add_legend()
    # plt.show()

    # drop useless columns
    train_df = train_df.drop(['Ticket', 'Cabin', 'Name', 'PassengerId', 'Embarked', 'Fare'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin', 'Name', 'PassengerId', 'Embarked', 'Fare'], axis=1)
    combine = [train_df, test_df]

    # change gender to numerical value
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    # cut age into ranges
    train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
    train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False)\
        .mean().sort_values(by='AgeBand', ascending=True)
    train_df = train_df.drop(['AgeBand'], axis=1)

    # create age range as numerical values
    for dataset in combine:
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4
    train_df.head()

# __________________________________________________DATA END___________________________________________________________

    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]

    # regression analysis
    logreg = LogisticRegression(solver="lbfgs")
    logreg.fit(X_train, Y_train)
    Y_pred_logreg = logreg.predict(test_df)
    acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
    print(acc_log)

    # Regression breakdown correlation
    # coeff_df = pd.DataFrame(train_df.columns.delete(0))
    # coeff_df.columns = ['Feature']
    # coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
    # coeff_df.sort_values(by='Correlation', ascending=False)

    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(test_df)
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    print(acc_random_forest)
    test_df['Predict'] = Y_pred
    test_df.to_csv(index=False, path_or_buf="../output.csv")

if __name__ == "__main__":
    train()
