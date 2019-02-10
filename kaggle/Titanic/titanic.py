import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as DTC

import os

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
result = test.copy()

category = ["Sex","Pclass"]
ans = ["Survived"]

dat = [train,test]
for i in dat:
    i["Sex"] = i["Sex"].map({"male":0,"female":1})


tr_c = train[category]
tr_r = train[ans]
test_c = test[category]

learnM = DTC()
learnM.fit(tr_c.values,tr_r.values)
res = learnM.predict(test_c.values)

sub = pd.DataFrame({"PassengerId": result.PassengerId, "Survived": res})
sub.to_csv("subtest.csv", index=False)