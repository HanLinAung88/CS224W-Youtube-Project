import snap
import numpy as np
from matplotlib import pyplot as plt
import csv

#hashes string to create unique integer id
def convertStrToUniqueInt(token):
    return int(str(hash(token))[:7])

#gets the id of the graph from a dictionary id
def getGraphId(video_id, dict_to_graph):
    return dict_to_graph[video_id]

#adds node to the graph and gives node a graph id
def addNodeToGraph(Graph, video_dict_id, dict_to_graph, current_graph_counter):
    if video_dict_id not in dict_to_graph:
        dict_to_graph[video_dict_id] = current_graph_counter
        Graph.AddNode(current_graph_counter)
        current_graph_counter += 1
    return current_graph_counter

#loads undirected graph based on video id dictionary
def load_graph_undirected(video_dict_list):
    Graph = snap.TUNGraph.New()
    dict_to_graph = {}
    current_graph_counter = 0

    for video_node in video_dict_list:
        video_dict_id = video_node['video_id']
        current_graph_counter = addNodeToGraph(Graph, video_dict_id, dict_to_graph, current_graph_counter)
        chosen_graph_id = dict_to_graph[video_dict_id]

        related_dict_ids = video_node['related_ids']
        for related_dict_id in related_dict_ids:
            current_graph_counter = addNodeToGraph(Graph, related_dict_id, dict_to_graph, current_graph_counter)
            related_graph_id = dict_to_graph[related_dict_id]
            if not Graph.IsEdge(chosen_graph_id, related_graph_id):
                Graph.AddEdge(chosen_graph_id, related_graph_id)

    return Graph

#loads the youtube dataset file and constructs a dictionary
def load_file(fname):
    fieldnames = ['video_id', 'uploader', 'age', 'category', 'length', 'views', 'rate', 'ratings', 'comments', 'related_ids']
    video_dict_list = []
    with open(fname) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            new_video_node = {}
            if(len(row) >= len(fieldnames)):
                for i in range(len(fieldnames) - 1):
                    new_video_node[fieldnames[i]] = row[i]
                related_video_list = row[len(fieldnames)-1 : ]
                new_video_node['related_ids'] = related_video_list
                video_dict_list.append(new_video_node)
    return video_dict_list

#plots a histogram
def plot_hist(data, x_label, y_label):
    plt.hist(x=data, bins=20)
    plt.grid(axis='y')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()