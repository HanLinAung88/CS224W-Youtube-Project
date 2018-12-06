from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import rolx
import features
import numpy as np
import utils
import random

# loads in the data by running rolx
fname = './dataset/0222/0.txt'
fname_extended = './dataset/0222/1.txt'
G, dict_to_graph, graph_to_dict = rolx.load_graph_igraph(fname)
roles = 5
H, R = rolx.extract_rolx_roles(G, roles)
print(H.shape, R.shape)

X = []
y = []
pos_data = []
neg_data = []

# extracts data from rolx for features
adj_mat = G.get_adjacency()
feature_dict = features.get_features(fname, fname_extended)
H.tolist()
for row in range(adj_mat.shape[0]):
    H_row = np.array(H[row]).flatten()
    for col in range(adj_mat.shape[1]):
        H_total = np.array(H[col][0]).flatten() + H_row + feature_dict[(row, col)]
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
clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)

# makes predictions
predictions = clf.predict(X_test)
accuracy = np.mean(predictions == y_test)
print('Accuracy:', accuracy)
np.savetxt('dataset/results.txt', predictions)
