import numpy as np
import pandas as pd
import scipy.stats as sc
import scipy.spatial.distance as sp
import matplotlib.pyplot as plt

train_data = pd.read_csv("propublicaTrain.csv")
test_data = pd.read_csv("propublicaTest.csv")
attr = train_data.columns.drop("two_year_recid")

# Helper Functions
def gaussian(data):
    mu = data.mean(axis=0)
    var = (data-mu).T @ (data-mu) / data.shape[0]
    return mu, var

def bayes_prob(row, train, attr):
    p = 1
    for i in range(len(attr)):
        if row[i+2] in train.index:
            p = p * train.loc[row[i+2]][attr[i]]
        else:
            return 0
    return p

# Maximum Likelihood Estimator
def MLE(train_data):

    # get data
    train_data_m0 = train_data[train_data.two_year_recid==0].fillna(0)
    train_data_m1 = train_data[train_data.two_year_recid==1].fillna(0)
    test_data_m = test_data.copy()
    test_data_m["y"] = -1

    # drop linearly dependent columns
    attr_m = attr[:-1]

    # generate Gaussian for the two classes
    mu_0, var_0 = gaussian(train_data_m0[attr_m])
    mu_1, var_1 = gaussian(train_data_m1[attr_m])

    model_0 = sc.multivariate_normal(mean=mu_0, cov=var_0)
    model_1 = sc.multivariate_normal(mean=mu_1, cov=var_1)

    # testing
    for i in range(len(test_data_m.index)):
        p_0 = model_0.pdf(test_data_m[attr_m].iloc[i])
        p_1 = model_1.pdf(test_data_m[attr_m].iloc[i])

        if p_0 > p_1:
            test_data_m["y"].iloc[i] = 0
        else:
            test_data_m["y"].iloc[i] = 1

    return len(test_data_m[test_data_m["two_year_recid"] == test_data_m["y"]].index)/len(test_data_m.index)


# K Nearest Neighbors

def KNN(k, p, train_data):
    train_data_knn = train_data.copy()
    test_data_knn = test_data.copy()
    test_data_knn["y"] = -1

    for i in range(len(test_data_knn.index)):
        distance = sp.cdist(train_data_knn[attr], test_data_knn[attr][i:i+1], 'minkowski', p)
        neighbors = np.apply_along_axis(np.argpartition, 0, distance, k)[:k]
        test_data_knn.iloc[i]["y"] = train_data_knn.iloc[list(neighbors[:, 0]), :]["two_year_recid"].mode()[0]

    return len(test_data_knn[test_data_knn["two_year_recid"] == test_data_knn["y"]].index)/len(test_data_knn.index)



# Naive Bayes Classifier

def NB(train_data):
    #split data
    train_data_b0 = train_data[train_data.two_year_recid==0].apply(pd.Series.value_counts).fillna(0)
    train_data_b1 = train_data[train_data.two_year_recid==1].apply(pd.Series.value_counts).fillna(0)

    #get counts and adjust for proportion
    counts = train_data.two_year_recid.value_counts()
    train_data_b0 = train_data_b0/counts[0]
    train_data_b1 = train_data_b1/counts[1]
    counts = counts/counts.sum()

    #get test data
    test_data_bayes = test_data.copy()
    test_data_bayes["y"] = -1

    #classification
    for row in test_data.itertuples():
        p_0 = bayes_prob(row, train_data_b0, attr) * counts[0]
        p_1 = bayes_prob(row, train_data_b1, attr) * counts[1]

        #give label
        if p_0 > p_1:
            test_data_bayes["y"].loc[row[0]] = 0
        else:
            test_data_bayes["y"].loc[row[0]] = 1

    return len(test_data_bayes[test_data_bayes["two_year_recid"] == test_data_bayes["y"]].index)/len(test_data_bayes.index)


