from unittest import result
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


noise_std = 0.01


def import_data():
    train = pd.read_csv("Data/train.csv")
    test = pd.read_csv("Data/test.csv")
    # print(train.head())

    return train, test


def show_data_size(train, test):
    plt.bar(range(2), (train.shape[0], test.shape[0]), align="center", alpha=0.8)
    plt.xticks(range(2), ("train", "test"))
    plt.ylabel("Number of data")
    plt.title("Data Samples")
    plt.show()


def show_column_distribution(data):
    plt.figure(figsize=(15, 15))
    for i in range(5):
        for j in range(5):
            plt.subplot(5, 5, 5 * i + j + 1)
            plt.hist(data[str(5 * i + j)], bins=100)
            plt.title("Column " + str(5 * i + j))
    plt.show()


def check_mean_std(data_X):
    print(data_X.mean().sum() / 300)
    print(data_X.std().sum() / 300)


def main():
    train, test = import_data()
    # show_data_size(train, test)
    # show_column_distribution(train)
    # show_column_distribution(test)

    train_X = train.drop(["id", "target"], axis=1)
    train_y = train["target"]
    test_X = test.drop(columns="id")
    # print(train_X.head())

    # check_mean_std(train_X)
    # check_mean_std(test_X)

    data = RobustScaler().fit_transform(np.concatenate((train_X, test_X)))
    train_X = data[:250]
    # train_X += np.random.normal(0, noise_std, train_X.shape)
    test_X = data[250:]

    clf = LogisticRegression(
        class_weight="balanced", solver="liblinear", penalty="l1", C=0.1, max_iter=10000
    )
    clf.fit(train_X, train_y)
    # print(f"5-fold val score : {cross_val_score(clf, train_X, train_y, cv=5)}")
    test_y = clf.predict_proba(test_X)
    submit = pd.read_csv("Data/sample_submission.csv")
    # result = []
    # for i in test_y[:, 1]:
    #     result.append(0 if i < 0.5 else 1)
    submit["target"] = test_y[:, 1]  # result
    submit.to_csv("submit.csv", index=False)


if __name__ == "__main__":
    main()
