import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import random


def data_load(root_folded="dataset/lfw-deepfunneled/lfw-deepfunneled/", if_show=False):
    dataset = []
    for path in glob.iglob(os.path.join(root_folded, "**", "*.jpg")):
        person = path.split("/")[-2]
        dataset.append({"person": person, "path": path})

    dataset = pd.DataFrame(dataset)
    dataset = dataset.groupby("person").filter(lambda x: len(x) > 10)
    print("Sample : /n", dataset.head(10))
    print("Size: ", len(dataset))

    if if_show:
        plt.figure(figsize=(20, 10))
        for i in range(20):
            idx = random.randint(0, len(dataset))
            img = plt.imread(dataset.path.iloc[idx])
            plt.subplot(4, 5, i + 1)
            plt.imshow(img)
            plt.title(dataset.person.iloc[idx])
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()

    return dataset


if __name__ == "__main__":
    data_load(if_show=True)
