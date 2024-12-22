import numpy as np
from sklearn.datasets import fetch_openml

if __name__ == "__main__":
    X,y = fetch_openml('mnist_784', version=1, return_X_y=True)
    y = y.astype(int)
    X = X/255.

    #Save the data to a file
    np.savez('mnist.npz', X=X, y=y)
    