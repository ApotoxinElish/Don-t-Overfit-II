import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedKFold

filte = [
    "0",
    "4",
    "9",
    "13",
    "14",
    "16",
    "17",
    "24",
    "26",
    "29",
    "30",
    "33",
    "39",
    "42",
    "43",
    "45",
    "48",
    "50",
    "62",
    "63",
    "65",
    "70",
    "73",
    "76",
    "80",
    "82",
    "91",
    "99",
    "104",
    "105",
    "108",
    "110",
    "114",
    "117",
    "119",
    "127",
    "129",
    "132",
    "133",
    "134",
    "135",
    "141",
    "147",
    "150",
    "151",
    "160",
    "164",
    "165",
    "166",
    "176",
    "180",
    "183",
    "189",
    "193",
    "194",
    "199",
    "201",
    "209",
    "217",
    "220",
    "225",
    "226",
    "228",
    "230",
    "231",
    "234",
    "235",
    "237",
    "239",
    "241",
    "242",
    "253",
    "258",
    "272",
    "275",
    "277",
    "279",
    "281",
    "282",
    "285",
    "292",
    "295",
    "298",
    "299",
]


def main():
    data = pd.read_csv("Data/train.csv")
    # print(data.iloc[:, 2:])
    X = data.iloc[:, 2:]
    # a = np.random.normal(0, 0.01, X.shape)
    # print(a)
    # X += a
    # print(X.shape)
    # return
    y = data["target"]
    # print(y)
    # skb = SelectKBest(f_classif, k=300)
    estimator = SVR(kernel="linear")
    # estimator = Lasso(alpha=0.031, tol=0.01, warm_start=True, selection="random")
    skb = RFECV(
        estimator,
        cv=StratifiedKFold(4),
        scoring="accuracy",
        min_features_to_select=1,
    )
    X_new = X[filte]  # skb.fit_transform(X, y)
    # print(skb.get_feature_names_out())
    print(X_new.shape)
    # return
    logreg_clf = LogisticRegression(solver="liblinear", random_state=0)
    param_grid = {
        "class_weight": ["balanced", None],
        "penalty": ["l1", "l2"],
        "C": [0.1, 0.5, 1.0, 1.5, 2.0],
    }
    clf = GridSearchCV(
        estimator=logreg_clf,
        param_grid=param_grid,
        # scoring="accuracy",
        # verbose=1,
        # n_jobs=-1,
        # cv=35,
    )

    # clf = MLPClassifier(
    #     solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(100, 10), random_state=1
    # )

    clf.fit(X_new, y)
    test = pd.read_csv("Data/test.csv")
    x_test = test.iloc[:, 1:]
    x_test = x_test[filte]  # [skb.get_feature_names_out()]
    r = clf.predict(x_test)
    # print(len(r))
    write = {"id": [i for i in range(250, 20000)], "target": r}
    df = pd.DataFrame(write)
    print(df)
    df.to_csv("test.csv", index=False)


if __name__ == "__main__":
    main()
