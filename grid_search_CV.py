from sklearn.model_selection import GridSearchCV
from dataloader import data_load

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import cv2
import numpy as np
from decompositions import decomposition_sklearn

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def grid_search_svm():
    dataset = data_load()
    train, test = train_test_split(dataset, test_size=0.1, random_state=0)
    print("Train:", len(train))
    print("Test:", len(test))

    clf = decomposition_sklearn(method_name="PCA")

    train_x = []
    train_y = []

    person_mapping = dict()
    name_idx = 0

    for idx in tqdm(range(len(train))):
        path = train.path.iloc[idx]
        name = train.person.iloc[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        X_decomposition = clf.fit_transform(img)

        X_decomposition = np.ravel(X_decomposition)

        if name in person_mapping:
            Y_train = person_mapping[name]
        else:
            Y_train = name_idx
            person_mapping[name] = name_idx
            name_idx += 1

        train_y.append(Y_train)
        train_x.append(X_decomposition)

    param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}

    svm_grid = GridSearchCV(SVC(), param_grid, cv=5)
    svm_grid.fit(train_x, train_y)

    print("Best SVM hyperparameters for PCA:", svm_grid.best_params_)
    print("Best SVM accuracy for PCA:", svm_grid.best_score_)

    clf = decomposition_sklearn(method_name="NMF")

    train_x = []
    train_y = []

    person_mapping = dict()
    name_idx = 0

    for idx in tqdm(range(len(train))):
        path = train.path.iloc[idx]
        name = train.person.iloc[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        X_decomposition = clf.fit_transform(img)

        X_decomposition = np.ravel(X_decomposition)

        if name in person_mapping:
            Y_train = person_mapping[name]
        else:
            Y_train = name_idx
            person_mapping[name] = name_idx
            name_idx += 1

        train_y.append(Y_train)
        train_x.append(X_decomposition)

    svm_grid = GridSearchCV(SVC(), param_grid, cv=5)
    svm_grid.fit(train_x, train_y)

    print("Best SVM hyperparameters for NMF:", svm_grid.best_params_)
    print("Best SVM accuracy for NMF:", svm_grid.best_score_)


def grid_search_knn():
    dataset = data_load()
    train, test = train_test_split(dataset, test_size=0.1, random_state=0)
    print("Train:", len(train))
    print("Test:", len(test))

    clf = decomposition_sklearn(method_name="PCA")

    train_x = []
    train_y = []

    person_mapping = dict()
    name_idx = 0

    for idx in tqdm(range(len(train))):
        path = train.path.iloc[idx]
        name = train.person.iloc[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        X_decomposition = clf.fit_transform(img)

        X_decomposition = np.ravel(X_decomposition)

        if name in person_mapping:
            Y_train = person_mapping[name]
        else:
            Y_train = name_idx
            person_mapping[name] = name_idx
            name_idx += 1

        train_y.append(Y_train)
        train_x.append(X_decomposition)

    grid_params = {'n_neighbors': [5, 7, 9, 11, 13, 15],
                   'weights': ['uniform', 'distance'],
                   'metric': ['minkowski', 'euclidean', 'manhattan']}

    cv_grid = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5)
    cv_grid.fit(train_x, train_y)

    print("Best KNeighborsClassifier hyperparameters for PCA:", cv_grid.best_params_)
    print("Best KNeighborsClassifier accuracy for PCA:", cv_grid.best_score_)

    clf = decomposition_sklearn(method_name="NMF")

    train_x = []
    train_y = []

    person_mapping = dict()
    name_idx = 0

    for idx in tqdm(range(len(train))):
        path = train.path.iloc[idx]
        name = train.person.iloc[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        X_decomposition = clf.fit_transform(img)

        X_decomposition = np.ravel(X_decomposition)

        if name in person_mapping:
            Y_train = person_mapping[name]
        else:
            Y_train = name_idx
            person_mapping[name] = name_idx
            name_idx += 1

        train_y.append(Y_train)
        train_x.append(X_decomposition)

    cv_grid = GridSearchCV(KNeighborsClassifier, grid_params, cv=5)
    cv_grid.fit(train_x, train_y)

    print("Best KNeighborsClassifier hyperparameters for NMF:", cv_grid.best_params_)
    print("Best KNeighborsClassifier accuracy for NMF:", cv_grid.best_score_)


def grid_search_rf():
    dataset = data_load()
    train, test = train_test_split(dataset, test_size=0.1, random_state=0)
    print("Train:", len(train))
    print("Test:", len(test))

    clf = decomposition_sklearn(method_name="PCA")

    train_x = []
    train_y = []

    person_mapping = dict()
    name_idx = 0

    for idx in tqdm(range(len(train))):
        path = train.path.iloc[idx]
        name = train.person.iloc[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        X_decomposition = clf.fit_transform(img)

        X_decomposition = np.ravel(X_decomposition)

        if name in person_mapping:
            Y_train = person_mapping[name]
        else:
            Y_train = name_idx
            person_mapping[name] = name_idx
            name_idx += 1

        train_y.append(Y_train)
        train_x.append(X_decomposition)

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    param_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    cv_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    cv_grid.fit(train_x, train_y)

    print("Best RandomForestClassifier hyperparameters for PCA:", cv_grid.best_params_)
    print("Best RandomForestClassifier accuracy for PCA:", cv_grid.best_score_)

    clf = decomposition_sklearn(method_name="NMF")

    train_x = []
    train_y = []

    person_mapping = dict()
    name_idx = 0

    for idx in tqdm(range(len(train))):
        path = train.path.iloc[idx]
        name = train.person.iloc[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        X_decomposition = clf.fit_transform(img)

        X_decomposition = np.ravel(X_decomposition)

        if name in person_mapping:
            Y_train = person_mapping[name]
        else:
            Y_train = name_idx
            person_mapping[name] = name_idx
            name_idx += 1

        train_y.append(Y_train)
        train_x.append(X_decomposition)

    cv_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    cv_grid.fit(train_x, train_y)

    print("Best RandomForestClassifier hyperparameters for NMF:", cv_grid.best_params_)
    print("Best RandomForestClassifier accuracy for NMF:", cv_grid.best_score_)


if __name__ == "__main__":
    grid_search_svm()
    grid_search_rf()
    grid_search_knn()
