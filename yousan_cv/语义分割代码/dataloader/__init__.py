"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .pascal_voc import voc_Loader
from .neus_seg import neu_seg
from .lips import lips

datasets = {
    'pascal_aug': voc_Loader,
    'neu':neu_seg,
    'lips':lips,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)

# /home/aries/Tutorial/awesome_segmentation/output/models/