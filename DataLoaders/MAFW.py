import os

import torch
import torch.utils.data as data
from imblearn import over_sampling

from . import utils


CLASSES = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
    "contempt",
    "anxiety",
    "helplessness",
    "disappointment"
]

"""
NEUTRAL_DESCR = [
    "Neutral facial expressions are characterized by a lack of emotional expression, as if the person's face is in a resting state.",
    "The facial muscles are generally relaxed, creating a smooth and even appearance.",
    "The mouth is typically closed or slightly open, with the lips not turned up or down.",
    "The eyebrows are in a neutral position, not furrowed or raised, and the eyes are generally looking straight ahead or slightly down.",
    "While the face may not show any specific emotions, the expression can still convey a sense of attentiveness or alertness."
]
"""


class MAFW(data.Dataset):
    def __init__(
            self,
            root_path: str,
            transforms: callable = None,
            target_transform: callable = None,
            load_transform: callable = None,
            fold: int = None,
            split: str = None,
            label_type: str = None,
            caption: bool = True
    ):
        super().__init__()
        if fold not in list(range(1, 6)):
            raise ValueError
        if split not in ['train', 'test']:
            raise ValueError
        if label_type not in ['compound', 'single']:
            raise ValueError
        self.root_path = root_path
        self.annotation_path = os.path.join(*[
            self.root_path,
            'Train_Test_Set',
            label_type,
            'with_caption_new' if caption else 'no_caption_new',
            'set_{}'.format(fold),
            '{}.txt'.format(split)
        ])
        self.transforms = transforms
        self.target_transform = target_transform
        self.load_transform = load_transform
        self.data = self._make_dataset(
            self.annotation_path
        )

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

        video_path = os.path.join(*[self.root_path, 'data', 'faces', video_name])
        video = utils.load_frames(video_path, time_transform=self.load_transform)
        if self.transforms is not None:
            video = self.transforms(video)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return video, label, description

    @staticmethod
    def _make_dataset(
            annotation_path: str
    ) -> list:
        annotations = utils.load_annotation(annotation_path)
        dataset = []
        for idx, row in enumerate(annotations):
            row = [el.replace('\n', '') for el in row]
            if len(row) == 4:
                filename = row[0].split('.')[0]
                emotion = row[1].split('_')
                description = row[3]
            else:
                filename = row[0].split('.')[0]
                emotion = row[1].split('_')
                description = "-"
            label = [CLASSES.index(emo) for emo in emotion]
            sample = {
                'video': filename,
                'descr': description,
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
