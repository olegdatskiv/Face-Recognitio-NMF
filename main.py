from dataloader import data_load
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

import numpy as np
from decompositions import decomposition_sklearn, decomposition

from classifier_models import classifier_full, classifier_short_version


def test_pca_sklearn(if_full_model=False):
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

    test_x = []
    test_y = []

    for idx in tqdm(range(len(test))):
        path = test.path.iloc[idx]
        name = test.person.iloc[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        X_decomposition = clf.fit_transform(img)

        X_decomposition = np.ravel(X_decomposition)

        if name in person_mapping:
            Y_test = person_mapping[name]
        else:
            continue

        test_y.append(Y_test)
        test_x.append(X_decomposition)

    if if_full_model:
        classifier_full(train_x, train_y, test_x, test_y)
    else:
        classifier_short_version(train_x, train_y, test_x, test_y)


def test_pca(if_full_model=False):
    dataset = data_load()
    train, test = train_test_split(dataset, test_size=0.1, random_state=0)
    print("Train:", len(train))
    print("Test:", len(test))

    clf = decomposition(method_name="PCA")

    train_x = []
    train_y = []

    person_mapping = dict()
    name_idx = 0

    for idx in tqdm(range(len(train))):
        path = train.path.iloc[idx]
        name = train.person.iloc[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        clf.fit(img)
        X_decomposition = clf.transform(img)

        X_decomposition = np.ravel(X_decomposition)

        if name in person_mapping:
            Y_train = person_mapping[name]
        else:
            Y_train = name_idx
            person_mapping[name] = name_idx
            name_idx += 1

        train_y.append(Y_train)
        train_x.append(X_decomposition)

    test_x = []
    test_y = []

    for idx in tqdm(range(len(test))):
        path = test.path.iloc[idx]
        name = test.person.iloc[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        clf.fit(img)
        X_decomposition = clf.transform(img)

        X_decomposition = np.ravel(X_decomposition)

        if name in person_mapping:
            Y_test = person_mapping[name]
        else:
            continue

        test_y.append(Y_test)
        test_x.append(X_decomposition)

    if if_full_model:
        classifier_full(train_x, train_y, test_x, test_y)
    else:
        classifier_short_version(train_x, train_y, test_x, test_y)


def test_nmf_sklearn(if_full_model=False):
    dataset = data_load()
    train, test = train_test_split(dataset, test_size=0.1, random_state=0)
    print("Train:", len(train))
    print("Test:", len(test))

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

    test_x = []
    test_y = []

    for idx in tqdm(range(len(test))):
        path = test.path.iloc[idx]
        name = test.person.iloc[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        X_decomposition = clf.fit_transform(img)

        X_decomposition = np.ravel(X_decomposition)

        if name in person_mapping:
            Y_test = person_mapping[name]
        else:
            continue

        test_y.append(Y_test)
        test_x.append(X_decomposition)

    if if_full_model:
        classifier_full(train_x, train_y, test_x, test_y)
    else:
        classifier_short_version(train_x, train_y, test_x, test_y)


def test_nmf(if_full_model=False):
    dataset = data_load()
    train, test = train_test_split(dataset, test_size=0.1, random_state=0)
    print("Train:", len(train))
    print("Test:", len(test))

    clf = decomposition(method_name="NMF")

    train_x = []
    train_y = []

    person_mapping = dict()
    name_idx = 0

    for idx in tqdm(range(len(train))):
        path = train.path.iloc[idx]
        name = train.person.iloc[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        X_decomposition, _ = clf.nmf_fit(img)

        X_decomposition = np.ravel(X_decomposition)

        if name in person_mapping:
            Y_train = person_mapping[name]
        else:
            Y_train = name_idx
            person_mapping[name] = name_idx
            name_idx += 1

        train_y.append(Y_train)
        train_x.append(X_decomposition)

    test_x = []
    test_y = []

    for idx in tqdm(range(len(test))):
        path = test.path.iloc[idx]
        name = test.person.iloc[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        X_decomposition, _ = clf.nmf_fit(img)

        X_decomposition = np.ravel(X_decomposition)

        if name in person_mapping:
            Y_test = person_mapping[name]
        else:
            continue

        test_y.append(Y_test)
        test_x.append(X_decomposition)

    if if_full_model:
        classifier_full(train_x, train_y, test_x, test_y)
    else:
        classifier_short_version(train_x, train_y, test_x, test_y)


def main():
    print("Result of PCA using sklearn library /n")
    test_pca_sklearn()
    print("---------------------------------------")
    print("---------------------------------------")
    print("---------------------------------------")
    print("---------------------------------------")
    print("Result of PCA from scratch /n")
    test_pca()
    print("---------------------------------------")
    print("---------------------------------------")
    print("---------------------------------------")
    print("---------------------------------------")
    print("Result of NMF using sklearn library /n")
    test_nmf_sklearn()
    print("---------------------------------------")
    print("---------------------------------------")
    print("---------------------------------------")
    print("---------------------------------------")
    print("Result of NMF from scratch /n")
    test_nmf()


if __name__ == '__main__':
    main()
