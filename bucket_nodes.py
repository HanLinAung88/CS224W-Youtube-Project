import utils

#gets the distribution of views of all videos and plots histogram
def get_views_distribution(fname):
    video_dict_list = utils.load_file(fname)
    views = [int(video_node['views']) for video_node in video_dict_list]
    utils.plot_hist(views, 'View count', 'Frequency')

fname = './dataset/0.txt'
get_views_distribution(fname)
