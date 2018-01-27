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

    def __init__(self, data_path: str = "data", batch_size: int = 10, patch_size: int = 572, augment: bool = True,
                 cache_in_memory: bool = False):
        self.data_path = data_path
        self.augment = augment
        self.batch_size = batch_size
        self.patch_size = patch_size

        self.cache_in_memory = cache_in_memory
        self.cache_x = dict()
        self.cache_y = dict()

        self.grid_sizes = pd.read_csv(os.path.join(self.data_path, 'grid_sizes.csv'), index_col=0)
        self.training_image_ids = [f for f in os.listdir(os.path.join(self.data_path, "train_geojson_v3"))
                                   if os.path.isdir(os.path.join(os.path.join(self.data_path, "train_geojson_v3"), f))]

        self.preprocess()

    def preprocess(self):
        """
        Does required preprocessing to get images ready for training. Also caches the results for future use.
        """
        cache_folder = os.path.join(self.data_path, "cache")
        if not os.path.isdir(cache_folder):
            os.makedirs(cache_folder)

        for image_id in self.training_image_ids:
            cache_path = os.path.join(cache_folder, "train_{image_id}".format(image_id=image_id))
            img_width = None
            img_height = None

            if not os.path.isfile(cache_path + "_x.npy"):
                print("Caching image {image_id}".format(image_id=image_id))
                temp_data_x = self.read_image(image_id)
                img_width = temp_data_x.shape[0]
                img_height = temp_data_x.shape[1]
                np.save(cache_path + "_x", temp_data_x)

            if not os.path.isfile(cache_path + "_y.npy"):
                print("Caching ground truth {image_id}".format(image_id=image_id))

                # In case image is not loaded we have to load to get dimensions
                if img_width is None:
                    temp_image = self.read_image(image_id)
                    img_width = temp_image.shape[0]
                    img_height = temp_image.shape[1]

                temp_data_y = np.zeros((img_width, img_height, 10))
                polygons = self.get_ground_truth_polys(image_id)
                for z in range(10):
                    temp_data_y[:, :, z] = self.get_ground_truth_array(polygons, z + 1, (img_width, img_height))
                np.save(cache_path + "_y", temp_data_y)

            if self.cache_in_memory:
                self.cache_x[image_id] = temp_data_x
                self.cache_y[image_id] = temp_data_y

    def next(self, amount: int = None):
        """
        Returns next batch of training images
        Tuple(x_train, y_train)
        x_train is a numpy array of shape [w, h, c]
        y_train is a numpy array of shape [w, h]
        """

        if amount is None:
            amount = self.batch_size

        # Extract a random subset of images from training pool (batch size)
        training_image_ids = [random.choice(self.training_image_ids) for _ in range(amount)]
        x_train_batch = []
        y_train_batch = []

        for image_id in training_image_ids:
            if self.cache_in_memory:
                x_train_temp = self.cache_x[image_id]
                y_train_temp = self.cache_y[image_id]
            else:
                x_train_temp = np.load(os.path.join(self.data_path, "cache",
                                                    "train_{image_id}_x.npy".format(image_id=image_id)))
                y_train_temp = np.load(os.path.join(self.data_path, "cache",
                                                    "train_{image_id}_y.npy".format(image_id=image_id)))

            if x_train_temp.shape[:2] != y_train_temp.shape[:2]:
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
                # Rotate either 0, 90, 180 or 270 degrees
                num_rotations = np.random.randint(4)
                x_train_temp = np.rot90(x_train_temp, num_rotations)
                y_train_temp = np.rot90(y_train_temp, num_rotations)

                # Flip horizontal
                if np.random.choice([True, False]):
                    x_train_temp = np.fliplr(x_train_temp)
                    y_train_temp = np.fliplr(y_train_temp)

                # Flip vertical
                if np.random.choice([True, False]):
                    x_train_temp = np.flipud(x_train_temp)
                    y_train_temp = np.flipud(y_train_temp)

            x_train_batch.append(x_train_temp)
            y_train_batch.append(y_train_temp)

        return np.array(x_train_batch), np.array(y_train_batch)

    def generator(self):
        while 1:
            yield self.next(amount=1)

    def get_grid_size(self, image_number: str) -> Tuple[float, float]:
        """
        Returns the grid size of a specific image. Needed to scale some coords
        """
        return self.grid_sizes[image_number]

    def read_image(self, image_id: str, band: int = 3):
        """
        Reads a image number from specified band and stores the image in a numpy array
        """
        if band == 3:
            filename = os.path.join(self.data_path, "three_band", '{image_id}.tif'.format(image_id=image_id))
            raw_data = tifffile.imread(filename).transpose([1, 2, 0])
            image_data = scale_image_percentile(raw_data)
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

    def get_ground_truth_array(self, polygons, class_number: int, image_size: Tuple[int, int]):
        """
        Creates a array containing class for each pixel
        """
        w, h = image_size

        # White background
        img_mask = np.full((w, h), 0, np.uint8)

        exteriors = [np.array(poly.exterior.coords).round().astype(np.int32)
                     for poly in polygons[str(class_number)]]

        cv2.fillPoly(img_mask, exteriors, 1)

        # Some polygons have regions inside them which need to be excluded
        interiors = [np.array(pi.coords).round().astype(np.int32)
                     for poly in polygons[str(class_number)] for pi in poly.interiors]

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
