import utils
from matplotlib import pyplot as plt

#gets the distribution of views of all videos and plots histogram
def get_views_distribution(fname):
    video_dict_list = utils.load_file(fname)
    views = [int(video_node['views']) for video_node in video_dict_list]
    sorted_views = sorted(views)
    print(len(sorted_views))
    utils.plot_hist(sorted_views[:100], 'View count', 'Frequency')
    utils.plot_hist(sorted_views[100:200], 'View count', 'Frequency')

fname = './dataset/0.txt'
get_views_distribution(fname)
