import numpy as np

from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class LaSOTDataset(BaseDataset):
    """
    LaSOT test set consisting of 280 videos (see Protocol-II in the LaSOT paper).

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """

    def __init__(self, subset):
        super().__init__()
        self.base_path = self.env_settings.lasot_path
        self.sequence_list = self._get_sequence_list(subset)
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls, _ = self.sequence_list[i].split('-')
            clean_lst.append(cls)
        return clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        class_name = sequence_name.split('-')[0]
        anno_path = '{}/{}/{}/groundtruth.txt'.format(self.base_path, class_name, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        occlusion_label_path = '{}/{}/{}/full_occlusion.txt'.format(self.base_path, class_name, sequence_name)

        # Note: pandas backed seems super super slow for loading occlusion/oov masks
        full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        out_of_view_label_path = '{}/{}/{}/out_of_view.txt'.format(self.base_path, class_name, sequence_name)
        out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = '{}/{}/{}/img'.format(self.base_path, class_name, sequence_name)

        frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number) for frame_number in
                       range(1, ground_truth_rect.shape[0] + 1)]

        target_class = class_name
        return Sequence(sequence_name, frames_list, 'lasot', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, subset):
        if subset == 'testing':
            return [
                'airplane-1',
                'airplane-9',
                'airplane-13',
                'airplane-15',
                'basketball-1',
                'basketball-6',
                'basketball-7',
                'basketball-11',
                'bear-2',
                'bear-4',
                'bear-6',
                'bear-17',
                'bicycle-2',
                'bicycle-7',
                'bicycle-9',
                'bicycle-18',
                'bird-2',
                'bird-3',
                'bird-15',
                'bird-17',
                'boat-3',
                'boat-4',
                'boat-12',
                'boat-17',
                'book-3',
                'book-10',
                'book-11',
                'book-19',
                'bottle-1',
                'bottle-12',
                'bottle-14',
                'bottle-18',
                'bus-2',
                'bus-5',
                'bus-17',
                'bus-19',
                'car-2',
                'car-6',
                'car-9',
                'car-17',
                'cat-1',
                'cat-3',
                'cat-18',
                'cat-20',
                'cattle-2',
                'cattle-7',
                'cattle-12',
                'cattle-13',
                'spider-14',
                'spider-16',
                'spider-18',
                'spider-20',
                'coin-3',
                'coin-6',
                'coin-7',
                'coin-18',
                'crab-3',
                'crab-6',
                'crab-12',
                'crab-18',
                'surfboard-12',
                'surfboard-4',
                'surfboard-5',
                'surfboard-8',
                'cup-1',
                'cup-4',
                'cup-7',
                'cup-17',
                'deer-4',
                'deer-8',
                'deer-10',
                'deer-14',
                'dog-1',
                'dog-7',
                'dog-15',
                'dog-19',
                'guitar-3',
                'guitar-8',
                'guitar-10',
                'guitar-16',
                'person-1',
                'person-5',
                'person-10',
                'person-12',
                'pig-2',
                'pig-10',
                'pig-13',
                'pig-18',
                'rubicCube-1',
                'rubicCube-6',
                'rubicCube-14',
                'rubicCube-19',
                'swing-10',
                'swing-14',
                'swing-17',
                'swing-20',
                'drone-13',
                'drone-15',
                'drone-2',
                'drone-7',
                'pool-12',
                'pool-15',
                'pool-3',
                'pool-7',
                'rabbit-10',
                'rabbit-13',
                'rabbit-17',
                'rabbit-19',
                'racing-10',
                'racing-15',
                'racing-16',
                'racing-20',
                'robot-1',
                'robot-19',
                'robot-5',
                'robot-8',
                'sepia-13',
                'sepia-16',
                'sepia-6',
                'sepia-8',
                'sheep-3',
                'sheep-5',
                'sheep-7',
                'sheep-9',
                'skateboard-16',
                'skateboard-19',
                'skateboard-3',
                'skateboard-8',
                'tank-14',
                'tank-16',
                'tank-6',
                'tank-9',
                'tiger-12',
                'tiger-18',
                'tiger-4',
                'tiger-6',
                'train-1',
                'train-11',
                'train-20',
                'train-7',
                'truck-16',
                'truck-3',
                'truck-6',
                'truck-7',
                'turtle-16',
                'turtle-5',
                'turtle-8',
                'turtle-9',
                'umbrella-17',
                'umbrella-19',
                'umbrella-2',
                'umbrella-9',
                'yoyo-15',
                'yoyo-17',
                'yoyo-19',
                'yoyo-7',
                'zebra-10',
                'zebra-14',
                'zebra-16',
                'zebra-17',
                'elephant-1',
                'elephant-12',
                'elephant-16',
                'elephant-18',
                'goldfish-3',
                'goldfish-7',
                'goldfish-8',
                'goldfish-10',
                'hat-1',
                'hat-2',
                'hat-5',
                'hat-18',
                'kite-4',
                'kite-6',
                'kite-10',
                'kite-15',
                'motorcycle-1',
                'motorcycle-3',
                'motorcycle-9',
                'motorcycle-18',
                'mouse-1',
                'mouse-8',
                'mouse-9',
                'mouse-17',
                'flag-3',
                'flag-9',
                'flag-5',
                'flag-2',
                'frog-3',
                'frog-4',
                'frog-20',
                'frog-9',
                'gametarget-1',
                'gametarget-2',
                'gametarget-7',
                'gametarget-13',
                'hand-2',
                'hand-3',
                'hand-9',
                'hand-16',
                'helmet-5',
                'helmet-11',
                'helmet-19',
                'helmet-13',
                'licenseplate-6',
                'licenseplate-12',
                'licenseplate-13',
                'licenseplate-15',
                'electricfan-1',
                'electricfan-10',
                'electricfan-18',
                'electricfan-20',
                'chameleon-3',
                'chameleon-6',
                'chameleon-11',
                'chameleon-20',
                'crocodile-3',
                'crocodile-4',
                'crocodile-10',
                'crocodile-14',
                'gecko-1',
                'gecko-5',
                'gecko-16',
                'gecko-19',
                'fox-2',
                'fox-3',
                'fox-5',
                'fox-20',
                'giraffe-2',
                'giraffe-10',
                'giraffe-13',
                'giraffe-15',
                'gorilla-4',
                'gorilla-6',
                'gorilla-9',
                'gorilla-13',
                'hippo-1',
                'hippo-7',
                'hippo-9',
                'hippo-20',
                'horse-1',
                'horse-4',
                'horse-12',
                'horse-15',
                'kangaroo-2',
                'kangaroo-5',
                'kangaroo-11',
                'kangaroo-14',
                'leopard-1',
                'leopard-7',
                'leopard-16',
                'leopard-20',
                'lion-1',
                'lion-5',
                'lion-12',
                'lion-20',
                'lizard-1',
                'lizard-3',
                'lizard-6',
                'lizard-13',
                'microphone-2',
                'microphone-6',
                'microphone-14',
                'microphone-16',
                'monkey-3',
                'monkey-4',
                'monkey-9',
                'monkey-17',
                'shark-2',
                'shark-3',
                'shark-5',
                'shark-6',
                'squirrel-8',
                'squirrel-11',
                'squirrel-13',
                'squirrel-19',
                'volleyball-1',
                'volleyball-13',
                'volleyball-18',
                'volleyball-19',
            ]
        elif subset == 'extension':
            return [
                'atv-1',
                'atv-2',
                'atv-3',
                'atv-4',
                'atv-5',
                'atv-6',
                'atv-7',
                'atv-8',
                'atv-9',
                'atv-10',
                'badminton-1',
                'badminton-2',
                'badminton-3',
                'badminton-4',
                'badminton-5',
                'badminton-6',
                'badminton-7',
                'badminton-8',
                'badminton-9',
                'badminton-10',
                'cosplay-1',
                'cosplay-10',
                'cosplay-2',
                'cosplay-3',
                'cosplay-4',
                'cosplay-5',
                'cosplay-6',
                'cosplay-7',
                'cosplay-8',
                'cosplay-9',
                'dancingshoe-1',
                'dancingshoe-2',
                'dancingshoe-3',
                'dancingshoe-4',
                'dancingshoe-5',
                'dancingshoe-6',
                'dancingshoe-7',
                'dancingshoe-8',
                'dancingshoe-9',
                'dancingshoe-10',
                'footbag-1',
                'footbag-2',
                'footbag-3',
                'footbag-4',
                'footbag-5',
                'footbag-6',
                'footbag-7',
                'footbag-8',
                'footbag-9',
                'footbag-10',
                'frisbee-1',
                'frisbee-2',
                'frisbee-3',
                'frisbee-4',
                'frisbee-5',
                'frisbee-6',
                'frisbee-7',
                'frisbee-8',
                'frisbee-9',
                'frisbee-10',
                'jianzi-1',
                'jianzi-2',
                'jianzi-3',
                'jianzi-4',
                'jianzi-5',
                'jianzi-6',
                'jianzi-7',
                'jianzi-8',
                'jianzi-9',
                'jianzi-10',
                'lantern-1',
                'lantern-2',
                'lantern-3',
                'lantern-4',
                'lantern-5',
                'lantern-6',
                'lantern-7',
                'lantern-8',
                'lantern-9',
                'lantern-10',
                'misc-1',
                'misc-2',
                'misc-3',
                'misc-4',
                'misc-5',
                'misc-6',
                'misc-7',
                'misc-8',
                'misc-9',
                'misc-10',
                'opossum-1',
                'opossum-2',
                'opossum-3',
                'opossum-4',
                'opossum-5',
                'opossum-6',
                'opossum-7',
                'opossum-8',
                'opossum-9',
                'opossum-10',
                'paddle-1',
                'paddle-2',
                'paddle-3',
                'paddle-4',
                'paddle-5',
                'paddle-6',
                'paddle-7',
                'paddle-8',
                'paddle-9',
                'paddle-10',
                'raccoon-1',
                'raccoon-2',
                'raccoon-3',
                'raccoon-4',
                'raccoon-5',
                'raccoon-6',
                'raccoon-7',
                'raccoon-8',
                'raccoon-9',
                'raccoon-10',
                'rhino-1',
                'rhino-2',
                'rhino-3',
                'rhino-4',
                'rhino-5',
                'rhino-6',
                'rhino-7',
                'rhino-8',
                'rhino-9',
                'rhino-10',
                'skatingshoe-1',
                'skatingshoe-2',
                'skatingshoe-3',
                'skatingshoe-4',
                'skatingshoe-5',
                'skatingshoe-6',
                'skatingshoe-7',
                'skatingshoe-8',
                'skatingshoe-9',
                'skatingshoe-10',
                'wingsuit-1',
                'wingsuit-2',
                'wingsuit-3',
                'wingsuit-4',
                'wingsuit-5',
                'wingsuit-6',
                'wingsuit-7',
                'wingsuit-8',
                'wingsuit-9',
                'wingsuit-10'
            ]
