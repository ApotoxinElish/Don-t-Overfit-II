import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


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
    print("mean:", data_X.mean().sum() / 300)
    print("std:", data_X.std().sum() / 300)


def save_data(target):
    submission = pd.read_csv("Data/sample_submission.csv")
    # print(submit.head())
    submission["target"] = target
    submission.to_csv("submission.csv", index=False)


def robust_scaler(train_X, test_X):
    data = RobustScaler().fit_transform(np.concatenate((train_X, test_X)))
    train_X = data[:250]
    train_X += np.random.normal(0, 0.01, train_X.shape)
    test_X = data[250:]
    return train_X, test_X


def random_forest_classifier():
    clf = RandomForestClassifier(
        random_state=42,
        n_estimators=100,
        criterion="gini",
        class_weight="balanced",
        max_depth=9,
        min_samples_leaf=7,
    )

    return clf


def k_neighbors_classifier():
    clf = KNeighborsClassifier(
        metric="minkowski",
        metric_params=None,
        leaf_size=2,
        n_neighbors=3,
        p=2,
        weights="uniform",
    )

    return clf


def mlp_classifier():
    clf = MLPClassifier(
        solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(100, 2), random_state=1
    )

    return clf


def logistic_regression():
    clf = LogisticRegression(
        class_weight="balanced", solver="liblinear", penalty="l1", C=0.1, max_iter=10000
    )

    return clf


def main():
    train, test = import_data()
    show_data_size(train, test)
    show_column_distribution(train)
    show_column_distribution(test)

    train_X = train.drop(["id", "target"], axis=1)
    train_y = train["target"]
    test_X = test.drop(columns="id")
    # print(train_X.head())

    # check_mean_std(train_X)
    check_mean_std(test_X)

    train_X, test_X = robust_scaler(train_X, test_X)

    # clf = random_forest_classifier()
    # clf = k_neighbors_classifier()
    # clf = mlp_classifier()
    clf = logistic_regression()

    clf.fit(train_X, train_y)
    proba = clf.predict_proba(test_X)
    # print(proba)
    test_y = proba[:, 1]
    # test_y = clf.predict(test_X)

    save_data(test_y)


if __name__ == "__main__":
    main()
