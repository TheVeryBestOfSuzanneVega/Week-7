import csv
import numpy as np
import pandas as pd
import sys
from sklearn import svm
import sklearn.model_selection
import sklearn.linear_model
import sklearn.neighbors
import sklearn.tree
import sklearn.ensemble

#csv list
csv_list = []

#reading in csv
df = pd.read_csv(sys.argv[1], skiprows=1, names=["a","b","label"])

#scatterplot
#colors = np.where(df["label"]==1,'r','-')
#colors[df["label"]==0] = 'b'
#df.plot(kind='scatter',x='a',y='b',c=colors)
#plt.show()

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df[["a","b"]], df["label"], test_size=0.4, stratify = df["label"] )

#linear
linear = ["svm_linear"]
parameters_linear = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
svc = svm.SVC(kernel='linear')
clf = sklearn.model_selection.GridSearchCV(svc, parameters_linear, cv = 5)
clf.fit(X_train, y_train)
linear.append(clf.best_score_)
linear.append(clf.score(X_test,y_test))
csv_list.append(linear)

#polynomial
poly = ["svm_polynomial"]
parameters_poly = {'C': [0.1, 1, 3], 'degree': [4,5,6], 'gamma': [0.1,0.5]}
svc = svm.SVC(kernel='poly')
clf = sklearn.model_selection.GridSearchCV(svc, parameters_poly, cv = 5)
clf.fit(X_train, y_train)
poly.append(clf.best_score_)
poly.append(clf.score(X_test,y_test))
csv_list.append(poly)

#rbf
rbf = ["svm_rbf"]
parameters_rbf = {'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma': [0.1, 0.5, 1, 3, 6, 10]}
svc = svm.SVC(kernel='rbf')
clf = sklearn.model_selection.GridSearchCV(svc, parameters_rbf, cv = 5)
clf.fit(X_train, y_train)
rbf.append(clf.best_score_)
rbf.append(clf.score(X_test,y_test))
csv_list.append(rbf)

#logistic regression
lr_list = ["logistic"]
parameters_lr = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
lr = sklearn.linear_model.LogisticRegression()
clf = sklearn.model_selection.GridSearchCV(lr, parameters_lr, cv = 5)
clf.fit(X_train, y_train)
lr_list.append(clf.best_score_)
lr_list.append(clf.score(X_test,y_test))
csv_list.append(lr_list)

#k-nearest neighbors
knn_list = ["knn"]
parameters_knn = {'n_neighbors': range(1,50), 'leaf_size': range(5,60,5)}
knn = sklearn.neighbors.KNeighborsClassifier()
clf = sklearn.model_selection.GridSearchCV(knn, parameters_knn, cv = 5)
clf.fit(X_train, y_train)
knn_list.append(clf.best_score_)
knn_list.append(clf.score(X_test,y_test))
csv_list.append(knn_list)

#decision tree
dt_list = ["decision_tree"]
parameters_dt = {'max_depth': range(1,50), 'min_samples_split': range(2,10)}
dt = sklearn.tree.DecisionTreeClassifier()
clf = sklearn.model_selection.GridSearchCV(dt, parameters_dt, cv = 5)
clf.fit(X_train, y_train)
dt_list.append(clf.best_score_)
dt_list.append(clf.score(X_test,y_test))
csv_list.append(dt_list)

#random forest
rf_list = ["random_forest"]
parameters_rf = {'max_depth': range(1,50), 'min_samples_split': range(2,10)}
rf = sklearn.ensemble.RandomForestClassifier()
clf = sklearn.model_selection.GridSearchCV(rf, parameters_rf, cv = 5)
clf.fit(X_train, y_train)
rf_list.append(clf.best_score_)
rf_list.append(clf.score(X_test,y_test))
csv_list.append(rf_list)

with open(sys.argv[2], 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    for w in csv_list:
        wr.writerow(w)
