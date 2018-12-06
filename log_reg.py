from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pickle
import rolx
import features
import numpy as np
import utils
import random

def get_scores(train_pred, train_true, test_pred, test_true):
    train_accuracy = np.mean(train_pred == train_true)
    train_f1 =  f1_score(train_true, train_pred)
    test_accuracy = np.mean(test_pred == test_true)
    test_f1 =  f1_score(test_true, test_pred)

    print('Train f1 scores: ', train_f1)
    print('Training Accuracy:', train_accuracy)
    print('Test f1 scores: ', test_f1)
    print('Testing Accuracy:', test_accuracy)

# loads in the data by running rolx
fname = './dataset/0222/0.txt'
fname_extended = './dataset/0222/1.txt'
G, dict_to_graph, graph_to_dict = rolx.load_graph_igraph(fname)
roles = 5
H, R = rolx.extract_rolx_roles(G, roles)
print(H.shape, R.shape)

# np.save('rolx_features', H)
# H = np.load('rolx_features.npy')

X = []
y = []
pos_data = []
neg_data = []

# extracts data from rolx for features
adj_mat = G.get_adjacency()
# feature_dict = features.get_features(fname, fname_extended)
# with open('feature_dict.pkl', 'wb') as f:
# 	pickle.dump(feature_dict, f)
# feature_dict = pickle.load('feature_dict.pkl')

H.tolist()
for row in range(adj_mat.shape[0]):
    H_row = np.array(H[row]).flatten()
    for col in range(adj_mat.shape[1]):
        H_total = np.array(H[col][0]).flatten() + H_row
        # H_total = np.concatenate([np.array(H[col][0]).flatten(), H_row])

        # only add into data if feature vector is valid
        # if (row,col) in feature_dict: 
        #     features = feature_dict[(row,col)]
        #     H_total = np.concatenate([H_total, features]) 

        if adj_mat[row][col] > 0:
            pos_data.append((H_total, adj_mat[row][col]))
        else:
            neg_data.append((H_total, adj_mat[row][col]))

# creates positive and negative dataset for more uniform distribution of data
X = [pos_data[i][0] for i in range(len(pos_data))]
Y = [pos_data[i][1] for i in range(len(pos_data))]

random_indices = sorted(random.sample(range(len(neg_data)), len(X)))
X_neg = [neg_data[i][0] for i in random_indices]
Y_neg = [neg_data[i][1] for i in random_indices]

X.extend(X_neg)
Y.extend(Y_neg)

# runs training by splitting train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print 'done splitting'
clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
print 'finished logistic regression'
# makes predictions
train_predictions = clf.predict(X_train)
test_predictions = clf.predict(X_test)

get_scores(train_predictions, y_train, test_predictions, y_test)
np.savetxt('dataset/results.txt', test_predictions)

