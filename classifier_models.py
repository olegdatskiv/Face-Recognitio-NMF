from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np


def classifier_full(train_x, train_y, test_x, test_y):
    clf_classifier = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf_classifier.fit(np.array(train_x), np.array(test_x), np.array(train_y), np.array(test_y))

    print(models)


def classifier_short_version(train_x, train_y, test_x, test_y):
    # Train a Support Vector Machine (SVM) classifier
    svm = SVC(kernel='linear', C=1)
    svm.fit(train_x, train_y)

    # Train a k-Nearest Neighbors (k-NN) classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_x, train_y)

    # Train a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(train_x, train_y)


    # Evaluate the performance of the classifiers on the test set
    print("SVM accuracy:", svm.score(test_x, test_y))
    print("k-NN accuracy:", knn.score(test_x, test_y))
    print("Random Forest accuracy:", rf.score(test_x, test_y))
