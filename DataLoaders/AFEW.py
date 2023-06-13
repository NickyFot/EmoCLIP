import os
import glob

import torch
import torch.utils.data as data
from imblearn import over_sampling

from . import utils

CLASSES = [
    "happiness",
    "sadness",
    "neutral",
    "anger",
    "surprise",
    "disgust",
    "fear"
]
script_dir = os.path.dirname(__file__)


class AFEW(data.Dataset):
    def __init__(
            self,
            root_path: str,
            transforms: callable = None,
            target_transform: callable = None,
            load_transform: callable = None,
            split: str = None
    ):
        if split not in ['train', 'test']:
            raise ValueError
        self.root_path = root_path
        self.split = 'validation' if split == 'test' else split
        self.transforms = transforms
        self.target_transform = target_transform
        self.load_transform = load_transform
        self.data = self._make_dataset()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.labels = [x['label'] for x in self.data]
        self.indices = list(range(0, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (clip, target, video_idx)
        """
        sample = self.data[index]
        video_name = sample['video']
        label = sample['label']
        description = sample['descr']
        emotion = sample['emotion']

        video_path = os.path.join(*[self.root_path, 'AFEW_Face', self.split, emotion, video_name])

        video = utils.load_frames(video_path, time_transform=self.load_transform)
        if self.transforms is not None:
            video = self.transforms(video)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return video, label, description

    def _make_dataset(self) -> list:
        # eg AFEW/AFEW_Face/train/Anger/0_2_001343200/frame.png
        videos = list(glob.glob(os.path.join(*[self.root_path, 'AFEW_Face', self.split, '*', '*'])))
        dataset = []
        for idx, row in enumerate(videos):
            row = row.split('/')
            video_idx = row[-1]
            emotion = row[-2]
            label = CLASSES.index(emotion.lower())

            description = ' '
            sample = {
                'video': video_idx,
                'descr': description,
                'emotion': emotion,
                'label': label
            }
            dataset.append(sample)
        return dataset

    def resample(self):
        sampler = over_sampling.RandomOverSampler()
        idx = torch.arange(len(self.data)).reshape(-1, 1)
        y = torch.tensor([sample['label'] for sample in self.data]).reshape(-1, 1)
        idx, _ = sampler.fit_resample(idx, y)
        idx = idx.reshape(-1)
        data = [self.data[i] for i in idx]
        self.data = data
