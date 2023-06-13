import json
import yaml
from pathlib import Path
import torch
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
import torchvision

from .MAFW import MAFW
from .FERV39K import FERV39K
from .DFEW import DFEW
from .AFEW import AFEW
from .utils import video_collate
from .utils import TemporalDownSample
from .utils import RandomSequence

from .MAFW import CLASSES as MAFW_CLASSES
from .FERV39K import CLASSES as FERV_CLASSES
from .DFEW import CLASSES as DFEW_CLASSES
from .AFEW import CLASSES as AFEW_CLASSES

CLASSES = list()
CLASS_DESCRIPTION = list()


def set_classinfo(cnf):
    class_descr = yaml.safe_load(Path('DataLoaders/class_descriptions.yml').read_text())
    if cnf.dataset_name == 'MAFW':
        CLASSES.extend(MAFW_CLASSES)
    elif cnf.dataset_name == 'FERV39K':
        CLASSES.extend(FERV_CLASSES)
    elif cnf.dataset_name == 'DFEW':
        CLASSES.extend(DFEW_CLASSES)
    elif cnf.dataset_name == 'AFEW':
        CLASSES.extend(AFEW_CLASSES)
    for cls in CLASSES:
        CLASS_DESCRIPTION.append(class_descr[cls])


def get_loaders(cnf, **kwargs):
    set_classinfo(cnf)
    if cnf.dataset_name == 'MAFW':
        return get_mafw_loaders(cnf)
    elif cnf.dataset_name == 'FERV39K':
        return get_ferv39k_loaders(cnf)
    elif cnf.dataset_name == 'DFEW':
        return get_dfew_loaders(cnf)
    elif cnf.dataset_name == 'AFEW':
        return get_afew_loaders(cnf)
    else:
        raise NotImplemented


def get_dfew_loaders(cnf):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
            torchvision.transforms.Resize(
                size=224,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                max_size=None,
                antialias=None
            ),
            torchvision.transforms.CenterCrop(size=(224, 224)),
            # torchvision.transforms.ColorJitter(brightness=0.5, hue=.2),
            torchvision.transforms.RandomRotation(6),
            torchvision.transforms.RandomHorizontalFlip()
        ]
    )

    load_transform = torchvision.transforms.Compose(
        [
            #RandomRoll(),
            TemporalDownSample(cnf.downsample),
            RandomSequence(cnf.clip_len, on_load=True)
        ]
    )
    dfew_train = DFEW(
        root_path=cnf.dataset_root,
        transforms=transforms,
        target_transform=None,
        load_transform=load_transform,
        split='train',
        fold=cnf.fold
    )
    if cnf.resample:
        dfew_train.resample()
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
            torchvision.transforms.Resize(
                size=224,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                max_size=None,
                antialias=None
            ),
            torchvision.transforms.CenterCrop(size=(224, 224))
        ]
    )
    dfew_test = DFEW(
        root_path=cnf.dataset_root,
        transforms=transforms,
        target_transform=None,
        load_transform=load_transform,
        split='test',
        fold=cnf.fold
    )
    if cnf.DDP:
        train_sampler = DistributedSampler(dfew_train)
        test_sampler = DistributedSampler(dfew_test)
    else:
        train_sampler = None
        test_sampler = None

    # print(len(train_sampler))
    trainloader = data.DataLoader(
        dfew_train,
        batch_size=cnf.batch_size,
        collate_fn=video_collate,
        num_workers=2,
        drop_last=True,
        sampler=train_sampler
    )
    testloader = data.DataLoader(
        dfew_test,
        batch_size=cnf.batch_size,
        collate_fn=video_collate,
        num_workers=2,
        sampler=test_sampler
    )

    print('Train Samples: {}, Test samples: {}'.format(len(dfew_train), len(dfew_test)))
    return trainloader, testloader


def get_afew_loaders(cnf):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
            torchvision.transforms.Resize(
                size=224,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                max_size=None,
                antialias=None
            ),
            torchvision.transforms.CenterCrop(size=(224, 224)),
            # torchvision.transforms.ColorJitter(brightness=0.5, hue=.2),
            torchvision.transforms.RandomRotation(6),
            torchvision.transforms.RandomHorizontalFlip()
        ]
    )

    load_transform = torchvision.transforms.Compose(
        [
            #RandomRoll(),
            TemporalDownSample(cnf.downsample),
            RandomSequence(cnf.clip_len, on_load=True)
        ]
    )
    afew_train = AFEW(
        root_path=cnf.dataset_root,
        transforms=transforms,
        target_transform=None,
        load_transform=load_transform,
        split='train'
    )
    if cnf.resample:
        afew_train.resample()
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
            torchvision.transforms.Resize(
                size=224,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                max_size=None,
                antialias=None
            ),
            torchvision.transforms.CenterCrop(size=(224, 224))
        ]
    )
    afew_test = AFEW(
        root_path=cnf.dataset_root,
        transforms=transforms,
        target_transform=None,
        load_transform=load_transform,
        split='test'
    )
    if cnf.DDP:
        train_sampler = DistributedSampler(afew_train)
        test_sampler = DistributedSampler(afew_test)
    else:
        train_sampler = None
        test_sampler = None

    # print(len(train_sampler))
    trainloader = data.DataLoader(
        afew_train,
        batch_size=cnf.batch_size,
        collate_fn=video_collate,
        num_workers=2,
        drop_last=True,
        sampler=train_sampler
    )
    testloader = data.DataLoader(
        afew_test,
        batch_size=cnf.batch_size,
        collate_fn=video_collate,
        num_workers=2,
        sampler=test_sampler
    )

    print('Train Samples: {}, Test samples: {}'.format(len(afew_train), len(afew_test)))
    return trainloader, testloader


def get_ferv39k_loaders(cnf):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
            torchvision.transforms.Resize(
                size=224,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                max_size=None,
                antialias=None
            ),
            torchvision.transforms.CenterCrop(size=(224, 224)),
            # torchvision.transforms.ColorJitter(brightness=0.5, hue=.2),
            torchvision.transforms.RandomRotation(6),
            torchvision.transforms.RandomHorizontalFlip()
        ]
    )

    load_transform = torchvision.transforms.Compose(
        [
            #RandomRoll(),
            TemporalDownSample(cnf.downsample),
            RandomSequence(cnf.clip_len, on_load=True)
        ]
    )
    ferv_train = FERV39K(
        root_path=cnf.dataset_root,
        transforms=transforms,
        target_transform=None,
        load_transform=load_transform,
        split='train'
    )
    if cnf.resample:
        ferv_train.resample()
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
            torchvision.transforms.Resize(
                size=224,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                max_size=None,
                antialias=None
            ),
            torchvision.transforms.CenterCrop(size=(224, 224))
        ]
    )
    ferv_test = FERV39K(
        root_path=cnf.dataset_root,
        transforms=transforms,
        target_transform=None,
        load_transform=load_transform,
        split='test'
    )
    if cnf.DDP:
        train_sampler = DistributedSampler(ferv_train)
        test_sampler = DistributedSampler(ferv_test)
    else:
        train_sampler = None
        test_sampler = None

    # print(len(train_sampler))
    trainloader = data.DataLoader(
        ferv_train,
        batch_size=cnf.batch_size,
        collate_fn=video_collate,
        num_workers=2,
        drop_last=True,
        sampler=train_sampler
    )
    testloader = data.DataLoader(
        ferv_test,
        batch_size=cnf.batch_size,
        collate_fn=video_collate,
        num_workers=2,
        sampler=test_sampler
    )

    print('Train Samples: {}, Test samples: {}'.format(len(ferv_train), len(ferv_test)))
    return trainloader, testloader


def get_mafw_loaders(cnf):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
            torchvision.transforms.Resize(
                size=224,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                max_size=None,
                antialias=None
            ),
            torchvision.transforms.CenterCrop(size=(224, 224)),
            # torchvision.transforms.ColorJitter(0.2, 0.2, 0.2),
            # torchvision.transforms.ColorJitter(brightness=0.5),
            torchvision.transforms.RandomRotation(6),
            torchvision.transforms.RandomHorizontalFlip()
        ]
    )

    load_transform = torchvision.transforms.Compose(
        [
            #RandomRoll(),
            TemporalDownSample(cnf.downsample),
            RandomSequence(cnf.clip_len, on_load=True)
        ]
    )
    mafw_train = MAFW(
        root_path=cnf.dataset_root,
        transforms=transforms,
        target_transform=None,
        load_transform=load_transform,
        fold=cnf.fold,
        split='train',
        label_type='single'
    )
    if cnf.resample:
        mafw_train.resample()
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
            torchvision.transforms.Resize(
                size=224,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                max_size=None,
                antialias=None
            ),
            torchvision.transforms.CenterCrop(size=(224, 224))
        ]
    )
    mafw_test = MAFW(
        root_path=cnf.dataset_root,
        transforms=transforms,
        target_transform=None,
        load_transform=load_transform,
        fold=cnf.fold,
        split='test',
        label_type='single',
        caption=False
    )
    if cnf.DDP:
        train_sampler = DistributedSampler(mafw_train)
        test_sampler = DistributedSampler(mafw_test)
    else:
        train_sampler = None
        test_sampler = None

    # print(len(train_sampler))
    trainloader = data.DataLoader(
        mafw_train,
        batch_size=cnf.batch_size,
        collate_fn=video_collate,
        # shuffle=True,
        num_workers=2,
        drop_last=True,
        sampler=train_sampler
    )
    testloader = data.DataLoader(
        mafw_test,
        batch_size=cnf.batch_size,
        collate_fn=video_collate,
        # shuffle=True,
        num_workers=2,
        sampler=test_sampler
    )

    print('Train Samples: {}, Test samples: {}'.format(len(mafw_train), len(mafw_test)))
    return trainloader, testloader

