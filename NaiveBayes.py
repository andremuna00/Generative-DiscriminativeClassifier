import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

# implement Naive Bayes Classifier here without using any library
class NaiveBayes:
    def __init__(self):
        self.alphas = np.zeros((10, 784))
        self.betas = np.zeros((10, 784))
        self.priors = np.zeros(10)

    def fit(self, X, y):
        self.X = X
        self.y = y
        for i in range(10):
            # seleziono i dati con label i
            X_i = self.X[self.y == i]
            # calcolo la media e la varianza
            mu_i = np.mean(X_i, axis=0)
            sigma_i = np.var(X_i, axis=0)
            # represent mean as a picture 32*32 and display it
            # img = mu_i.reshape(28,28)
            # save image
            # plt.imsave('/NaiveBayesLearning/mean'+str(i)+'.png', img, cmap='gray')

            #calcolo alpha e beta
            for j in range(784):
                k = (mu_i[j] * (1 - mu_i[j]) - sigma_i[j] + 0.001 )/( sigma_i[j] + 0.001) 
                self.alphas[i][j] = 1+ mu_i[j] * k
                self.betas[i][j] = 1+(1 - mu_i[j]) * k
            #calcolo la prior
            self.priors[i] = X_i.shape[0] / self.X.shape[0]


    def predict(self, X):
        len = X.shape[0]
        y_pred = np.zeros(len)
        for i in range(len):
            # calcolo la probabilità di ogni classe
            probs = np.zeros(10)
            for j in range(10):
                prod = 1
                for k in range(784):
                    if(self.alphas[j][k]>0 and self.betas[j][k]>0):
                        p = math.gamma(self.alphas[j][k]+self.betas[j][k])
                        p1 = math.gamma(self.alphas[j][k])
                        p2 = math.gamma(self.betas[j][k])
                        cost = p/(p1*p2)
                        x = X[i][k]
                        if X[i][k]==0:
                            x=0.0001
                        elif X[i][k]==1:
                            x=0.9999

                        ex = (x**(self.alphas[j][k]-1)) * ((1-x)**(self.betas[j][k]-1))
                        value = cost*ex
                        prod *= value
                probs[j] = self.priors[j] * prod
            # prendo la classe con probabilità maggiore
            y_pred[i] = np.argmax(probs)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        x = accuracy_score(y, y_pred)
        return x

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    


if __name__ == "__main__":
    # read data from file
    data = np.load('mnist.npz')
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.float32)
    print("Data loaded")

    # Split the test set
    X_train = X[:60000]
    y_train = y[:60000]
    X_test = X[-10000:]
    y_test = y[-10000:]

    # Test the model
    model = NaiveBayes()
    start_fit = time.time()
    model.fit(X_train, y_train)
    end_fit = time.time()

    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()

    print("Training Time: ", end_fit - start_fit)
    print("Testing Time: ", end - start)
    print("Accuracy: ", accuracy_score(y_test, y_pred))

    # Save the time in a txt file
    with open('NaiveBayes_infos.txt', 'w') as f:
        f.write("Training Time: " + str(end_fit - start_fit))
        f.write("\nTesting Time: " + str(end - start))
        f.write("\nAccuracy: " + str(accuracy_score(y_test, y_pred)))

    #plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    #save plot
    plt.savefig('NaiveBayes_confusion_matrix.png')

    plt.show()