import matplotlib.pyplot as plt

from lib.test.analysis.extract_results import extract_results

plt.rcParams['figure.figsize'] = [8, 8]

import _init_paths
from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'lasot'
trackers.extend(trackerlist(name='mlgt', parameter_name='vitb_256_ep300', dataset_name=dataset_name,
                            run_ids=None, display_name='vitb_256_ep300'))
# trackers.extend(trackerlist(name='mlgt', parameter_name='vitl_320_ep300', dataset_name='NOTU',
#                             run_ids=None, display_name='vitl_320_ep300'))

dataset = get_dataset('lasot')
print_results(trackers, dataset, 'lasot', merge_results=True, plot_types=('success', 'prec','norm_prec'), avist=False)
# dataset = get_dataset('uav')
# print_results(trackers, dataset, 'UAV123', merge_results=True, plot_types=('success', 'prec'), avist=False)
# dataset = get_dataset('avist')
# print_results(trackers, dataset, 'AVisT', merge_results=True, plot_types=('success'), avist=True)
# plot_results(trackers, dataset, 'lasot', merge_results=True, plot_types=('success','prec','norm_prec'), avist=False,
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)

# print_per_sequence_results(trackers, dataset, 'lasot', merge_results=True, filter_criteria=None)

# extract_results(trackers, dataset, 'lasot', skip_missing_seq=False, plot_bin_gap=0.05, avist=False)
