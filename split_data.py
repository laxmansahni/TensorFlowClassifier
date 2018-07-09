import numpy as np               # 1
import pandas as pd

df = pd.read_csv("voice.csv", header=0)        # 2

labels = (df["label"] == "male").values * 1    # 3
labels = labels.reshape(-1, 1)                 # 4

del df["label"]                  # 5
data = df.values

# 6
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                       test_size=0.3, random_state=123456)

np.save("X_train.npy", X_train)  # 7
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
