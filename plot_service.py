import matplotlib.pyplot as plt
import numpy as np


def plot_sample(X_train_orig, Y_train_orig):

    # Example of a picture
    index = 0
    plt.imshow(X_train_orig[index])
    plt.show()
    print("y = " + str(np.squeeze(Y_train_orig[:, index])))
