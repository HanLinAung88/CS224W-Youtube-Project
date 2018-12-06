import utils
import snap
from collections import defaultdict

# using this video dict list, extract proximity features from the descriptions of nodes
def proximity_features(video_dict_list):
	pass

def aggregate_features(video_dict_list, vid1, vid2, fields):
	feature_size = len(fields)*2
	# if both exist in the video dict list, calculate the aggregates
	if vid1 in video_dict_list and vid2 in video_dict_list:
		features = []
		# for each field, aggregate by taking sum and difference
		for field in fields:
			num1 = video_dict_list[vid1][field]
			num2 = video_dict_list[vid2][field]
			features.append(float(num1)+float(num2))
			features.append(float(num1)-float(num2))
	else:
		features = [None for i in range(feature_size)]
	return features

def common_neighbors_similarity(n1, n2, neighbors):
	return len(neighbors[n1] & neighbors[n2])

def jaccard_similarity(n1, n2, neighbors):
	intersection = neighbors[n1] & neighbors[n2]
	union = neighbors[n1] | neighbors[n2]
	return float(len(intersection)) / len(union)

def get_neighbors(G):
	neighbors = defaultdict(set)
	for node in G.Nodes():
		for n in node.GetOutEdges():
			neighbors[node.GetId()].add(n)
	return neighbors

def topological_features(n1, n2, neighbors):
	c = common_neighbors_similarity(n1, n2, neighbors)
	j = jaccard_similarity(n1, n2, neighbors)
	features = [c,j]
	return features

def extract_features(G, video_dict_list, graph_to_dict):
	# loop through pairs of nodes in the graph
	fields = ['views', 'ratings', 'comments']
	# the total size for a default value, aggregate feature size
	neighbors = get_neighbors(G)
	feature_dict = {} 

	for node1 in graph_to_dict:
		for node2 in graph_to_dict:
			vid1 = graph_to_dict[node1]
			vid2 = graph_to_dict[node2]

			agg_features = aggregate_features(video_dict_list, vid1, vid2, fields)
			topo_features = topological_features(node1, node2, neighbors)

			# combine all the features	
			features = agg_features + topo_features
			feature_dict[(node1, node2)] = features

	return feature_dict


def get_features(fname, fname_extended):
	video_dict_list = utils.load_file(fname)
	video_dict_list_extended = utils.load_file(fname_extended)
	G, dict_to_graph, graph_to_dict = utils.load_graph_undirected(video_dict_list)
	# use the extended video dict list as it will contain more information on the videos from the crawl
	feature_dict = extract_features(G, video_dict_list_extended, graph_to_dict)
	return feature_dict

# test to see if everything is working
def main():
	fname = './dataset/0222/0.txt'
	fname_extended = './dataset/0222/1.txt'
	feature_dict = get_features(fname, fname_extended)
	print(len(feature_dict))

if __name__ == "__main__":
	main()
