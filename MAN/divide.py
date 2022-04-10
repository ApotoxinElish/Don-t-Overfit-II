import pandas as pd
import numpy as np


data = pd.read_csv("Data/train.csv", header=0)
data = data.dropna()

pos = int((data["target"] == 1).sum())
neg = int((data["target"] == 0).sum())


def divide(stepNum):
    gaps = []
    for title in list(data.columns):
        if title == "id" or title == "target":
            continue
        max = -1000
        min = 1000
        for index, row in data.iterrows():
            if row[title] > max:
                max = row[title]
            if row[title] < min:
                min = row[title]
        step = (max - min) / stepNum
        maxGap = 0
        for i in range(stepNum):
            negLes = 0
            posLes = 0

            separator = min + i * step
            for index, row in data.iterrows():
                if row["target"] == 0 and row[title] < separator:
                    negLes += 1
                if row["target"] == 1 and row[title] < separator:
                    posLes += 1

            posPercent = posLes / pos
            negPercent = negLes / neg
            if maxGap < abs(posPercent - negPercent):
                maxGap = abs(posPercent - negPercent)
        gaps.append(maxGap)
    gaps = [str(x) for x in gaps]
    f = open("divide_" + str(stepNum) + ".txt", "w")
    f.write(",".join(gaps))
    f.close()


divide(50)
