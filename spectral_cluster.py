import snap
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import utils

def get_adjacency_matrix(Graph):
    '''
    This function builds the adjacency matrix of a
    given graph and return it as a numpy array
    '''
    ##########################################################################
    num_nodes = Graph.GetNodes()
    adj_mat = np.zeros((num_nodes, num_nodes))

    for curr_node in Graph.Nodes():
        num_neighbors = curr_node.GetDeg()
        curr_node_id = curr_node.GetId()
        if(num_neighbors > 0):
            for neighbor_index in range(num_neighbors):
                neighbor_node_id = curr_node.GetNbrNId(neighbor_index)
                adj_mat[curr_node_id, neighbor_node_id] = 1

    return adj_mat
    ##########################################################################

def get_sparse_degree_matrix(Graph):
    '''
    This function builds the degree matrix of a
    given graph and return it as a numpy array
    '''
    ##########################################################################
    num_nodes = Graph.GetNodes()
    sparse_deg_matrix = np.zeros(num_nodes)
    for curr_node in Graph.Nodes():
        curr_node_id = curr_node.GetId()
        num_neighbors = curr_node.GetDeg()
        sparse_deg_matrix[curr_node_id] = num_neighbors

    return np.diag(sparse_deg_matrix)
    ##########################################################################

def normalized_cut_minimization(Graph):
    '''
    Normalized cut minimizaton algorithm
    '''
    A = get_adjacency_matrix(Graph)
    D = get_sparse_degree_matrix(Graph)
    ##########################################################################
    A = get_adjacency_matrix(Graph)
    D = get_sparse_degree_matrix(Graph)
    L = D - A
    L_tilde = np.linalg.inv(np.sqrt(D)).dot(L).dot(np.linalg.inv(np.sqrt(D)))
    eigenvals, eigenvecs = np.linalg.eigh(L_tilde)
    print "eigenvalues are ", eigenvals[:10]
    print "eigenvectors are", eigenvecs[:10]
    v = eigenvecs[:, 1]
    return np.linalg.inv(np.sqrt(D)).dot(v)
    ##########################################################################

def getQ_sum(Graph, c):
    '''
    helper function for modularity
    '''
    adj_mat = get_adjacency_matrix(Graph)
    Q_sum = 0
    sum_Aij = np.sum(adj_mat)
    for i in range(0, len(c)):
        curr_node = Graph.GetNI(c[i])
        for j in range(0, len(c)):
            curr_node_2 = Graph.GetNI(c[j])
            Q_sum += adj_mat[c[i], c[j]] - (curr_node.GetDeg() * curr_node_2.GetDeg())/float(sum_Aij)
    return Q_sum

def modularity(Graph, c1, c2):
    '''
    This function helps compute the modularity of a given cut
    defined by two sets c1 and c2. We would normally require sets c1 and c2
    to be disjoint and to include all nodes in Graph
    '''
    ##########################################################################
    Q_c1_sum = getQ_sum(Graph, c1)
    Q_c2_sum = getQ_sum(Graph, c2)
    adj_mat = get_adjacency_matrix(Graph)
    sum_Aij = np.sum(adj_mat)

    return (1.0/sum_Aij) * (Q_c1_sum + Q_c2_sum)
    ##########################################################################

def printMetrics(Graph, S, S_bar, name):
    print(name)
    print('Size of S: ', len(S))
    print('Size of S_bar: ', len(S_bar))
    print('Modularity: ', modularity(Graph, S, S_bar))

def splitSets(vec_x):
    set_positive = []
    set_negative = []

    for i in range(0, len(vec_x)):
        x = vec_x[i]
        if x > 0:
            set_positive.append(i)
        else:
            set_negative.append(i)
    return set_positive, set_negative

def generate_categories(video_dict_list, graph_to_dict, cluster):
    categories = defaultdict(int)
    for nid in cluster:
        video_id = utils.getVideoId(nid, graph_to_dict)
        try: 
            categories[(video_dict_list[video_id]['category'])] += 1
        except:
            continue
    return categories

def spectral_cluster(fname, fname_extended):
    '''
    Main spectral clustering workflow
    '''
    ##########################################################################
    video_dict_list_extended = utils.load_file(fname_extended)

    video_dict_list = utils.load_file(fname)
    youtube, dict_to_graph, graph_to_dict = utils.load_graph_undirected(video_dict_list)
    video_dict_list.update(video_dict_list_extended)

    x_youtube = normalized_cut_minimization(youtube)
    S, S_bar = splitSets(x_youtube)
    S_categories = generate_categories(video_dict_list, graph_to_dict, S)
    S_bar_categories = generate_categories(video_dict_list, graph_to_dict, S_bar)
    print(S_categories)
    print(S_bar_categories)

    printMetrics(youtube, S, S_bar, "Youtube spectral clustering")

fname = './dataset/0222/0.txt'
fname_extended = './dataset/0222/1.txt'
spectral_cluster(fname, fname_extended)
