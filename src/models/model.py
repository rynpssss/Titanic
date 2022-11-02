from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def logistic_regression(X_train, y_train, x_test):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(x_test)
    acc = round(logreg.score(X_train, y_train) * 100, 2)

    return acc


def support_vecotor_classification(X_train, Y_train, X_test):
    svc = SVC()
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    acc = round(svc.score(X_train, Y_train) * 100, 2)
    
    return acc


def k_neighbors_classifier(X_train, Y_train, X_test, n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors = n_neighbors)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    acc = round(knn.score(X_train, Y_train) * 100, 2)

    return acc


def random_forest(X_train, Y_train, X_test, n_estimators=100):
    random_forest = RandomForestClassifier(n_estimators=n_estimators)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    acc = round(random_forest.score(X_train, Y_train) * 100, 2)

    return acc
