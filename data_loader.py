import csv
import os
import random
from typing import Tuple, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.affinity
import shapely.wkt
import tifffile
import webcolors
from shapely.geometry import MultiPolygon

csv.field_size_limit(2**24)

LABEL_TO_CLASS = {
    'LARGE_BUILDING': 1,
    'RESIDENTIAL_BUILDING': 1,
    'NON_RESIDENTIAL_BUILDING': 1,
    'EXTRACTION_MINE': 1,
    'MISC_SMALL_STRUCTURE': 2,
    'MISC_SMALL_MANMADE_STRUCTURE': 2,
    'GOOD_ROADS': 3,
    'POOR_DIRT_CART_TRACK': 4,
    'FOOTPATH_TRAIL': 4,
    'WOODLAND': 5,
    'HEDGEROWS': 5,
    'GROUP_TREES': 5,
    'STANDALONE_TREES': 5,
    'CONTOUR_PLOUGHING_CROPLAND': 6,
    'ROW_CROP': 6,
    'FARM_ANIMALS_IN_FIELD': 6,
    'DEMARCATED_NON_CROP_FIELD': 6,
    'WATERWAY': 7,
    'STANDING_WATER': 8,
    'LARGE_VEHICLE': 9,
    'SMALL_VEHICLE': 10,
    'MOTORBIKE': 10
}

CLASS_TO_LABEL = {
    1: 'BUILDING',
    2: 'STRUCTURE',
    3: 'ROAD',
    4: 'TRAIL',
    5: 'TREES',
    6: 'FARMLAND',
    7: 'WATERWAY',
    8: 'STILL WATER',
    9: 'LARGE VEHICLE',
    10: 'SMALL VEHICLE'
}

COLOR_MAPPING = {
    1: "#aaaaaa",  # Buildings
    2: "#666666",  # Small structure
    3: "#b35806",  # Road
    4: "#dfc27d",  # Trail / Dirt road
    5: "#1b7837",  # Trees
    6: "#a6dba0",  # Farmland
    7: "#74add1",  # Waterway
    8: "#4575b4",  # Still water
    9: "#f46d43",  # Large vehicle
    10: "#d73027",  # Small vehicle
}

# Sort layers by which one is above the others
# We need this because a car might be on top of a road or a building might be on top of farmland
ZORDER = {
    1: 5,
    2: 5,
    3: 4,
    4: 1,
    5: 3,
    6: 2,
    7: 7,
    8: 8,
    9: 9,
    10: 10
}


def scale_image_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix


class Generator:
    """
    Class responsible for generating batches of data to train on
    """

    def __init__(self, data_path: str = "data", batch_size: int = 10, patch_size: int = 572, augment: bool = True):
        self.data_path = data_path
        self.augment = augment
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.cache = dict()

        # TODO: Pre-fetch image sizes
        self.grid_sizes = pd.read_csv(os.path.join(self.data_path, 'grid_sizes.csv'), index_col=0)
        self.training_image_ids = [f for f in os.listdir(os.path.join(self.data_path, "train_geojson_v3"))
                                   if os.path.isdir(os.path.join(os.path.join(self.data_path, "train_geojson_v3"), f))]

    def next(self) -> Tuple[any, any]:
        """
        Returns next batch of training images
        Tuple(x_train, y_train)
        x_train is a numpy array of shape [w, h, c]
        y_train is a numpy array of shape [w, h]
        """
        # Extract a random subset of images from training pool (batch size)
        training_image_ids = random.sample(self.training_image_ids, self.batch_size)
        x_train_batch = []
        y_train_batch = []

        for training_image_id in training_image_ids:
            x_train_temp = self.read_image(training_image_id)
            y_train_temp = self.get_ground_truth_array(training_image_id)

            if x_train_temp.shape[:2] != y_train_temp.shape:
                raise Exception("Shape of data does not match shape of ground truth")

            # Crop to patch size
            start_index_width = np.random.randint(0, x_train_temp.shape[0] - self.patch_size)
            start_index_height = np.random.randint(0, x_train_temp.shape[1] - self.patch_size)

            x_train_temp = x_train_temp[start_index_width:start_index_width + self.patch_size,
                                        start_index_height:start_index_height + self.patch_size]

            y_train_temp = y_train_temp[start_index_width:start_index_width + self.patch_size,
                                        start_index_height:start_index_height + self.patch_size]

            # Augment
            if self.augment:
                num_rotations = np.random.randint(4)
                flip_vertical = np.random.choice([True, False])
                flip_horizontal = np.random.choice([True, False])

                x_train_temp = np.rot90(x_train_temp, num_rotations)
                y_train_temp = np.rot90(y_train_temp, num_rotations)

                if flip_horizontal:
                    x_train_temp = np.fliplr(x_train_temp)
                    y_train_temp = np.fliplr(y_train_temp)

                if flip_vertical:
                    x_train_temp = np.flipud(x_train_temp)
                    y_train_temp = np.flipud(y_train_temp)

                print(f"{num_rotations} Vert {flip_vertical} Hori {flip_horizontal}")

            x_train_batch.append(x_train_temp)
            y_train_batch.append(y_train_temp)

        return x_train_batch, y_train_batch

    def get_grid_size(self, image_number: str) -> Tuple[float, float]:
        """
        Returns the grid size of a specific image. Needed to scale some coords
        """
        return self.grid_sizes[image_number]

    def read_image(self, image_number: str, band: int = 3):
        """
        Reads a image number from specified band and stores the image in a numpy array
        """
        if band == 3:
            if f"{image_number}_{band}" in self.cache:
                return self.cache[f"{image_number}_{band}"]

            filename = os.path.join(self.data_path, "three_band", f'{image_number}.tif')
            raw_data = tifffile.imread(filename).transpose([1, 2, 0])
            image_data = scale_image_percentile(raw_data)
            self.cache[f"{image_number}_{band}"] = image_data
            return image_data
        else:
            raise Exception("Only 3-band is implemented")

    def save_image(self, image_number: str, filename: str, band: int = 3, view: bool = False) -> None:
        """
        Saves a image number from specified band and saves it to filename
        Overwrites existing files without warning
        """
        image_data = self.read_image(image_number, band)
        if view:
            plt.imshow(image_data)
        else:
            plt.imsave(filename, image_data)

    def save_overlay(self, image_number: str, filename: str, view: bool = False) -> None:
        """
        Saves a image number from specified band and saves it to filename
        Overwrites existing files without warning
        """
        train_mask = self.mask_for_polygons(image_number)
        if view:
            plt.imshow(train_mask)
        else:
            plt.imsave(filename, train_mask)

    def scale_coords(self, img_size: Tuple[int, int], image_number: str) -> Tuple[float, float]:
        """
        Get a scaling factor needed to scale polygons to same size as image
        """
        x_max, y_min = self.grid_sizes.loc[image_number][['Xmax', 'Ymin']]
        h, w = img_size
        w_ = w * (w / (w + 1))
        h_ = h * (h / (h + 1))
        return w_ / x_max, h_ / y_min

    def get_ground_truth_polys(self, image_number: str) -> Dict[str, MultiPolygon]:
        """
        Get a list of polygons sorted by class for the selected image.
        Scaled to match image.
        """
        train_polygons = dict()
        for _im_id, _poly_type, _poly in csv.reader(open(os.path.join(self.data_path, 'train_wkt_v4.csv'))):
            if _im_id == image_number:
                train_polygons[_poly_type] = shapely.wkt.loads(_poly)

        x_scale, y_scale = self.scale_coords(self.read_image(image_number).shape[:2], image_number)

        train_polygons_scaled = dict()
        for key, train_polygon in train_polygons.items():
            train_polygons_scaled[key] = shapely.affinity.scale(train_polygon, xfact=x_scale, yfact=y_scale,
                                                                origin=(0, 0, 0))

        return train_polygons_scaled

    def get_ground_truth_array(self, image_number: str):
        """
        Creates a array containing class for each pixel
        """
        w, h = self.read_image(image_number).shape[:2]
        polygons = self.get_ground_truth_polys(image_number)

        # White background
        img_mask = np.full((w, h), 0, np.uint8)

        # Sort polygons by Z-order
        for cls, _ in sorted(ZORDER.items(), key=lambda x: x[1]):
            exteriors = [np.array(poly.exterior.coords).round().astype(np.int32)
                         for poly in polygons[str(cls)]]

            cv2.fillPoly(img_mask, exteriors, int(cls))

            # Some polygons have regions inside them which need to be excluded
            interiors = [np.array(pi.coords).round().astype(np.int32)
                         for poly in polygons[str(cls)] for pi in poly.interiors]

            cv2.fillPoly(img_mask, interiors, 0)

        return img_mask

    def mask_for_polygons(self, image_number: str):
        """
        Create a color mask of classes with the same size as original image
        """
        w, h = self.read_image(image_number).shape[:2]
        polygons = self.get_ground_truth_polys(image_number)

        # White background
        img_mask = np.full((w, h, 3), 255, np.uint8)

        # Sort polygons by Z-order
        for cls, _ in sorted(ZORDER.items(), key=lambda x: x[1]):
            exteriors = [np.array(poly.exterior.coords).round().astype(np.int32) for poly in polygons[str(cls)]]
            cv2.fillPoly(img_mask, exteriors, webcolors.hex_to_rgb(COLOR_MAPPING[int(cls)]))

            # Some polygons have regions inside them which need to be excluded
            interiors = [np.array(pi.coords).round().astype(np.int32)
                         for poly in polygons[str(cls)] for pi in poly.interiors]
            cv2.fillPoly(img_mask, interiors, (255, 255, 255))

        return img_mask


if __name__ == "__main__":
    generator = Generator()
    x_train, y_train = generator.next()
