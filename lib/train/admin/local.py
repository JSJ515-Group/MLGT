import os


class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = os.path.expanduser(
            '~') + f'/track/code/MLGT'  # Base directory for saving network checkpoints
        self.tensorboard_dir = os.path.expanduser(
            '~') + f'/track/code/MLGT/tensorboard'  # Directory for tensorboard files
        self.pretrained_networks = os.path.expanduser('~') + f'/track/code/MLGT/pretrained_networks'
        self.lasot_dir = 'D:\\BaiduNetdiskDownload\\LaSOT\\LaSOT\\LaSOTBenchmark'
        self.got10k_dir = 'C:\\Users\\cmm\\Desktop\\dataset\\GOT-10k\\train'
        self.trackingnet_dir = 'E:\\trackingnet'
        self.coco_dir = 'C:\\Users\\cmm\\Desktop\\dataset\\coco'
        self.got10k_val_dir = 'C:\\Users\\cmm\\Desktop\\dataset\\GOT-10k\\val'

        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
