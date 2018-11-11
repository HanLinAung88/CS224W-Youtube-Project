import utils
from collections import defaultdict
import numpy as np

#partitions the dictionary list based on the value to be partitioned on (such as category)
def partition_dict(video_dict_list, partition_key, numeric_bins = None):
    partitioned_dict = defaultdict(list)
    for dict_id in video_dict_list:
        partition_value = video_dict_list[dict_id][partition_key]
        if numeric_bins is not None:
            for i in range(len(numeric_bins)):
                if float(partition_value) <= numeric_bins[i]:
                    partition_value = numeric_bins[i]
                    break
        partitioned_dict[partition_value].append(dict_id)
    return partitioned_dict

# gets the distribution of views of all videos and plots histogram
# returns a dictionary {views: [list of video_id]}
def get_views_distribution(fname):
    video_dict_list = utils.load_file(fname)
    views = [int(video_dict_list[video_id]['views']) for video_id in video_dict_list]
    sorted_views = sorted(views)
    n, bins, patches = utils.plot_hist(sorted_views[:100], 'View count', 'Frequency')
    n2, bins2, patches2 = utils.plot_hist(sorted_views[100:200], 'View count', 'Frequency')
    return partition_dict(video_dict_list, 'views', bins)

# gets the distribution of categories of all videos
# returns a dictionary {video_category: [list of video_id]}
def get_categories_distribution(fname):
    video_dict_list = utils.load_file(fname)
    categories = partition_dict(video_dict_list, 'category')
    X = []
    Y = []
    for key in categories:
        X.append(key)
        Y.append(len(categories[key]))
    utils.plot_barGraph([X, Y], "Categories", "Frequency")
    return categories

#gets the distribution of ratings of all videos and plots histogram
# returns a dictionary {ratings: [list of video_id]}
def get_ratings_distribution(fname):
    video_dict_list = utils.load_file(fname)
    ratings = [float(video_dict_list[video_id]['ratings']) for video_id in video_dict_list]
    sorted_ratings = sorted(ratings)
    n, bins, patches = utils.plot_hist(sorted_ratings, 'Ratings', 'Frequency')
    return partition_dict(video_dict_list, 'ratings', bins)


fname = './dataset/0.txt'
views_dict = get_views_distribution(fname)
categories_dict = get_categories_distribution(fname)
ratings_dict = get_ratings_distribution(fname)

# print(views_dict)
# print(categories_dict)
# print(ratings_dict)
