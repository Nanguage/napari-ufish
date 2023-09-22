"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

from skimage.io import imread
import os.path as osp

HERE = osp.abspath(osp.dirname(__file__))


def make_sample_data():
    """Generates an image"""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image
    sample1_path = osp.join(HERE, "sample_data/sample1.tif")
    sample1 = imread(sample1_path)
    return [(sample1, {})]
