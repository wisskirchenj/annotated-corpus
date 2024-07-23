import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=182)
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)
# load data to test model
data = pd.read_csv('../data/hyperskill-dataset-97997779.txt')
target_predict = model.predict(data)
# count occurrences of each class in the prediction
unique, counts = np.unique(target_predict, return_counts=True)
print(unique, counts)
