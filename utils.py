import snap
import numpy as np
from matplotlib import pyplot as plt
import csv
import networkx as nx

#hashes string to create unique integer id
def convertStrToUniqueInt(token):
    return int(str(hash(token))[:7])

#gets the id of the graph from a dictionary id
def getGraphId(video_id, dict_to_graph):
    return dict_to_graph[video_id]

#gets the id of the graph from a dictionary id
def getVideoId(node_id, graph_to_dict):
    return graph_to_dict[node_id]

#adds node to the graph and gives node a graph id
def addNodeToGraph(Graph, video_dict_id, dict_to_graph, graph_to_dict, current_graph_counter):
    if video_dict_id not in dict_to_graph:
        dict_to_graph[video_dict_id] = current_graph_counter
        graph_to_dict[current_graph_counter] = video_dict_id
        Graph.AddNode(current_graph_counter)
        current_graph_counter += 1
    return current_graph_counter

#loads undirected graph based on video id dictionary
def load_graph_undirected(video_dict_list):
    Graph = snap.TUNGraph.New()
    dict_to_graph = {}
    graph_to_dict = {}
    current_graph_counter = 0

    for video_node in video_dict_list.values():
        video_dict_id = video_node['video_id']
        current_graph_counter = addNodeToGraph(Graph, video_dict_id, dict_to_graph, graph_to_dict, current_graph_counter)
        chosen_graph_id = dict_to_graph[video_dict_id]

        related_dict_ids = video_node['related_ids']
        for related_dict_id in related_dict_ids:
            current_graph_counter = addNodeToGraph(Graph, related_dict_id, dict_to_graph, graph_to_dict, current_graph_counter)
            related_graph_id = dict_to_graph[related_dict_id]
            if not Graph.IsEdge(chosen_graph_id, related_graph_id):
                Graph.AddEdge(chosen_graph_id, related_graph_id)
    snap.SaveEdgeList(Graph, 'youtube_graph.txt')
    return Graph, dict_to_graph, graph_to_dict

#loads the youtube dataset file and constructs a dictionary
def load_file(fname):
    fieldnames = ['video_id', 'uploader', 'age', 'category', 'length', 'views', 'rate', 'ratings', 'comments', 'related_ids']
    video_dict_list = {}
    with open(fname) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            new_video_node = {}
            if(len(row) >= len(fieldnames)):
                for i in range(len(fieldnames) - 1):
                    new_video_node[fieldnames[i]] = row[i]
                related_video_list = row[len(fieldnames)-1 : ]
                new_video_node['related_ids'] = related_video_list
                video_dict_list[row[0]] = new_video_node
    return video_dict_list

#plots a bar graph
def plot_barGraph(data, x_label, y_label):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    names = data[0]
    values = data[1]
    plt.bar(range(len(names)), values, align='center')
    plt.xticks(range(len(names)), names)
    plt.show()

#plots a histogram
def plot_hist(data, x_label, y_label):
    n, bins, patches = plt.hist(x=data, bins=100)
    plt.grid(axis='y')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    return n, bins, patches

#loads networkX graph
def load_graph_networkX(fname_edgelist):
    G = nx.read_edgelist(fname_edgelist)
    return G

#gets all the categories in the current dictionary
def get_all_categories(video_dict_list):
    categories = set()
    for video_id in video_dict_list:
        categories.add(video_dict_list[video_id]['category'])
    return list(categories)

#plots networkX graph with color and node_sizes
def plot_graph_networkX(G, graph_to_dict, video_dict_list):
    node_sizes = [10.0 if graph_to_dict[int(node)] in video_dict_list else 0.01 for node in G]
    categories = get_all_categories(video_dict_list)
    node_colors = []
    for node in G:
        if graph_to_dict[int(node)] in video_dict_list:
            node_colors.append((categories.index(video_dict_list[getVideoId(int(node), graph_to_dict)]['category']) + 1.0)/float(len(categories)))
        else:
            node_colors.append(0.0)
    nx.draw_kamada_kawai(G, with_labels = False, node_size = node_sizes, node_color = node_colors)
    plt.show()
    plt.savefig('youtube_graph.png')
