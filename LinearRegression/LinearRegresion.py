import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

savedModelFile = "studentmodel.pickle"

# Comment start to test pickle

best_acc = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test,y_test)
    print(acc)
    if acc > best_acc:
        best_acc = acc
        with open(savedModelFile,"wb") as f:
            pickle.dump(linear, f)

# Comment end to test pickle


pickle_in = open(savedModelFile, "rb")
linear = pickle.load(pickle_in)

coeff = linear.coef_

print(coeff)

predictions = linear.predict(x_test)

'''
for i in range(len(predictions)):
    print(predictions[i], y_test[i], "-->", x_test[i])
'''

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel(predict)
pyplot.show()
