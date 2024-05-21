import os

from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here

    settings.davis_dir = ''
    settings.got10k_path = 'C:\\Users\\cmm\\Desktop\\dataset\\GOT-10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = 'D:\\BaiduNetdiskDownload\\LaSOT\\LaSOT\\LaSOTBenchmark'  # Where tracking networks are stored
    # settings.lasot_path = 'D:\\BaiduNetdiskDownload\\LaSOT_Ext'
    settings.nfs_path = 'D:\\BaiduNetdiskDownload\\Nfs'
    settings.otb_path = 'D:\\BaiduNetdiskDownload\\OTB100'
    settings.prj_dir = 'C:\\Users\\cmm\\Desktop\\MLGT\\MLGT'
    settings.result_plot_path = 'C:\\Users\\cmm\\Desktop\\MLGT\\MLGT/output1/test/result_plots'
    settings.results_path = 'C:\\Users\\cmm\\Desktop\\MLGT\\MLGT/output1/test/tracking_results'  # Where to store tracking results
    settings.save_dir = 'C:\\Users\\cmm\\Desktop\\MLGT\\MLGT\\lib\\train\\output1'
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = 'E:\\TEST'
    settings.uav_path = 'D:\\BaiduNetdiskDownload\\UAV123\\UAV123\\Dataset_UAV123\\UAV123'
    settings.vot18_path = ''
    settings.vot22_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''
    settings.show_result = False
    return settings
