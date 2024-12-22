import numpy as np
import pandas as pd
import time

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class kNN:
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        #We need only to store the training data and the labels
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        #initialize the vector of predictions
        len = X.shape[0]
        y_pred = np.zeros(len)
        for i in range(len):
            #euclidean distance
            distances = np.sqrt(np.sum((X[i] - self.X_train)**2, axis=1))
            indexes = np.argsort(distances)
            best_k_idx = indexes[:self.k]
            #retrieve the labels of the k nearest points using the indexes found before
            k_nearest_labels = [self.y_train[i] for i in best_k_idx]
            #find the most frequent label in the k nearest points
            y_pred[i] = max(k_nearest_labels, key=k_nearest_labels.count)
        return y_pred
        
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self, deep=True):
        return {'k': self.k}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


if __name__ == "__main__":
    #read data from file
    data = np.load('mnist.npz')
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.float32)
    print("Data loaded")

    # Split the test set
    X_search = X[:10000]
    y_search = y[:10000]
    X_train = X[:60000]
    y_train = y[:60000]
    X_test = X[-10000:]
    y_test = y[-10000:]

    # 10 way cross validation
    param_grid = {
        'k': [1, 3, 5, 7, 9]
    }
    search = GridSearchCV(kNN(), param_grid, cv=10, n_jobs=16)
    start_search = time.time()
    search.fit(X_search, y_search)
    end_search = time.time()
    print("Search Time: ", end_search - start_search)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    #print mean scores for each parameter
    means = search.cv_results_['mean_test_score']
    stds = search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    #create model with best parameters
    model = kNN(k=search.best_params_['k'])
    # Fit the model
    start_fit = time.time()
    model.fit(X_train, y_train)
    end_fit = time.time()
    # Test the model
    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()

    print("Training Time: ", end_fit - start_fit)
    print("Testing Time: ", end - start)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    sorted(search.cv_results_.keys())

    # Print all search result in a csv file
    df = pd.DataFrame(search.cv_results_)
    df.to_csv('kNN_search_results.csv')
    # Save the time in a txt file
    with open('kNN_infos.txt', 'w') as f:
        f.write("Best parameter (CV score=%0.3f):" % search.best_score_)
        f.write(str(search.best_params_))
        f.write("\nSearch Time: " + str(end_search - start_search))
        f.write("\nTraining Time: " + str(end_fit - start_fit))
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
    plt.savefig('kNN_confusion_matrix.png')

    plt.show()




    