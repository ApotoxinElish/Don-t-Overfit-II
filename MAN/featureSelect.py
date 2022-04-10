import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso
from sklearn.datasets import make_classification


def rawSelect(divideNum, minGap):
    f = open("divide_" + str(divideNum) + ".txt", "r")
    gaps = f.read().split(",")
    gaps = [float(x) for x in gaps]

    selected = []
    for i in range(len(gaps)):
        if gaps[i] >= minGap:
            selected.append(str(i))
    return selected


features = rawSelect(50, 0.12)

trainDf = pd.read_csv("Data/train.csv")
testDf = pd.read_csv("Data/test.csv").drop(["id"], axis=1)

X = trainDf.drop(["id", "target"], axis=1)[features]
y = trainDf["target"]
# y.loc[y['target'] == 1, 'target'] = int(1)
# y.loc[y['target'] == 0, 'target'] = int(0)

estimator = SVC(kernel="linear")
# estimator = Lasso(alpha=0.03, tol=0.01, warm_start=True, selection="random")
min_features_to_select = 1
rfecv = RFECV(
    estimator=estimator,
    cv=StratifiedKFold(4),
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
)
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)
print("features", list(rfecv.feature_names_in_))
print(list(rfecv.get_support(indices=True)))
