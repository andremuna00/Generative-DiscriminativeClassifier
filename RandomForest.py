import numpy as np
import pandas as pd
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #read data from file
    data = np.load('mnist.npz')
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.float32)
    print("Data loaded")
    
    # Test is the last 10000 samples
    X_search = X[:10000]
    y_search = y[:10000]
    X_train = X[:60000]
    y_train = y[:60000]
    X_test = X[-10000:]
    y_test = y[-10000:]

    # 10 way cross validation
    param_grid = {
        'n_estimators': [10, 100, 1000],
        'max_depth': [2, 10, 40, 100],
        'min_samples_leaf': [1, 2, 4], #minimum number of samples required to be at a leaf node.
        'min_samples_split': [2, 5, 10], #minimum number of samples required to split an internal node
    }

    
    search = GridSearchCV(RandomForestClassifier(), param_grid, cv=10, n_jobs=16)
    start_search = time.time()
    search.fit(X_search, y_search)
    end_search = time.time()
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    #print mean scores for each parameter
    means = search.cv_results_['mean_test_score']
    stds = search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    #create model with best parameters
    model = RandomForestClassifier(n_estimators=search.best_params_['n_estimators'], max_depth=search.best_params_['max_depth'], min_samples_leaf=search.best_params_['min_samples_leaf'], min_samples_split=search.best_params_['min_samples_split'])
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

    #print all search result in a csv file
    df = pd.DataFrame(search.cv_results_)
    df.to_csv('RandomForest_search_results.csv')
    #save the time of testing in a txt file
    with open('RandomForest_infos.txt', 'w') as f:
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
    plt.savefig('RandomForest_confusion_matrix.png')

    plt.show()
    
    