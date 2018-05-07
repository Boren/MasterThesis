import csv
import os
import random
from typing import Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import scipy
import shapely.affinity
import shapely.wkt
import tifffile
from numpy.lib.stride_tricks import as_strided
from skimage.transform import rescale
from shapely.geometry import MultiPolygon

from utils.visualize import ZORDER

csv.field_size_limit(2 ** 24)


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

    def __init__(self, data_path: str = "data", batch_size: int = 10,
                 patch_size: int = 572, augment: bool = True,
                 classes = range(8), channels=3):
        self.data_path = data_path
        self.augment = augment
        self.batch_size = batch_size
        self.patch_size = patch_size

        self.classes = classes
        self.channels = channels

        self.cache_x = dict()
        self.cache_y = dict()

        grid_sizes_file = os.path.join(self.data_path, 'grid_sizes.csv')
        self.grid_sizes = pd.read_csv(grid_sizes_file, index_col=0)

        self.training_image_ids = self.get_image_ids('train')
        self.validation_image_ids = self.get_image_ids('validation')

        self.all_image_ids = [os.path.splitext(f)[0] for f in os.listdir(
            os.path.join(self.data_path, 'three_band'))]
        self.preprocess()

    def get_image_ids(self, type: str):
        folder = os.path.join(self.data_path, '{}_geojson'.format(type))
        return [f for f in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, f))]

    def preprocess(self):
        """
        Performs required preprocessing to get images ready for training.
        Also caches the results for future use.
        """
        cache_folder = os.path.join(self.data_path, "cache")
        if not os.path.isdir(cache_folder):
            os.makedirs(cache_folder)

        for image_id in self.training_image_ids + self.validation_image_ids:
            cache_path = os.path.join(cache_folder, "{}".format(image_id))
            img_width = None
            img_height = None

            if not os.path.isfile(cache_path + "_x.npy") and os.path.isfile(os.path.join(self.data_path, 'three_band', '{}.tif'.format(image_id))):
                print("Caching image {}".format(image_id))
                temp_data_x = self.read_image(image_id)
                img_width = temp_data_x.shape[0]
                img_height = temp_data_x.shape[1]
                np.save(cache_path + "_x", temp_data_x)

            if not os.path.isfile(cache_path + "_M.npy") and os.path.isfile(os.path.join(self.data_path, 'sixteen_band', '{}_M.tif'.format(image_id))):
                print("Caching image {} - M band".format(image_id))

                # In case image is not loaded we have to load to get dimensions
                if img_width is None:
                    temp_image = self.read_image(image_id)
                    img_width = temp_image.shape[0]
                    img_height = temp_image.shape[1]

                temp_data_x = self.read_image(image_id, band="M")
                temp_data_x = self.reshape(temp_data_x, (img_width, img_height))
                np.save(cache_path + "_M", temp_data_x)

            if not os.path.isfile(cache_path + "_A.npy") and os.path.isfile(os.path.join(self.data_path, 'sixteen_band', '{}_A.tif'.format(image_id))):
                print("Caching image {} - A band".format(image_id))

                # In case image is not loaded we have to load to get dimensions
                if img_width is None:
                    temp_image = self.read_image(image_id)
                    img_width = temp_image.shape[0]
                    img_height = temp_image.shape[1]

                temp_data_x = self.read_image(image_id, band="A")
                temp_data_x = self.reshape(temp_data_x, (img_width, img_height))
                np.save(cache_path + "_A", temp_data_x)

            if not os.path.isfile(cache_path + "_y.npy") and image_id in self.training_image_ids + self.validation_image_ids:
                print("Caching ground truth {}".format(image_id))

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

    def next(self, amount: int = None, data_type: str = 'train',
             classes=range(8)):
        """
        Returns next batch of training images
        Tuple(x_train, y_train)
        x_train is a numpy array of shape [w, h, c]
        y_train is a numpy array of shape [w, h, num_classes]
        """

        if amount is None:
            amount = self.batch_size

        # Extract a random subset of images from training pool (batch size)
        if data_type == 'train':
            image_ids = np.random.choice(self.training_image_ids, amount, True)
        elif data_type == 'validation':
            image_ids = np.random.choice(self.validation_image_ids, amount, True)
        else:
            raise Exception("Not a valid dataset")

        x_train_batch = []
        y_train_batch = []

        for image_id in image_ids:
            if self.channels == 3:
                x_train_temp = np.load(os.path.join(self.data_path, "cache", "{}_x.npy".format(image_id)), mmap_mode='r+')
            elif self.channels == 8:
                x_train_temp = np.load(os.path.join(self.data_path, "cache", "{}_M.npy".format(image_id)), mmap_mode='r+')
            elif self.channels == 16:
                x_train_M = np.load(os.path.join(self.data_path, "cache", "{}_M.npy".format(image_id)), mmap_mode='r+')
                x_train_A = np.load(os.path.join(self.data_path, "cache", "{}_A.npy".format(image_id)), mmap_mode='r+')
                x_train_temp = np.concatenate((x_train_M, x_train_A), axis=2)
            else:
                raise Exception("Wrong number of channels")

            y_train_temp = np.load(os.path.join(self.data_path, "cache", "{}_y.npy".format(image_id)), mmap_mode='r+')

            if x_train_temp.shape[:2] != y_train_temp.shape[:2]:
                raise Exception("Shape of data does not match shape of ground truth. {} vs {}.".format(x_train_temp.shape, y_train_temp.shape))

            # Crop to patch size
            start_width = np.random.randint(0, x_train_temp.shape[0] - self.patch_size)
            start_height = np.random.randint(0, x_train_temp.shape[1] - self.patch_size)

            x_train_temp = x_train_temp[
                           start_width:start_width + self.patch_size,
                           start_height:start_height + self.patch_size]

            y_train_temp = y_train_temp[
                           start_width:start_width + self.patch_size,
                           start_height:start_height + self.patch_size]

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

        return np.array(x_train_batch), np.array(y_train_batch)[:,:,:,classes]

    @staticmethod
    def reshape(arr, shape):
        scaled = np.empty((shape[0], shape[1], arr.shape[2]))

        for i in range(arr.shape[2]):
            img = arr[:, :, i]
            scale = scipy.misc.imresize(img, shape)
            scaled[:, :, i] = scale

        return scaled


    def get_patch(self, image: str, x: int, y: int, width: int, height: int, channels: int = 3):
        if self.channels == 3:
            x_train = np.load(os.path.join(self.data_path, "cache", "{}_x.npy".format(image)), mmap_mode='r+')
        elif self.channels == 8:
            x_train = np.load(os.path.join(self.data_path, "cache", "{}_M.npy".format(image)), mmap_mode='r+')
        elif self.channels == 16:
            x_train_M = np.load(os.path.join(self.data_path, "cache", "{}_M.npy".format(image)), mmap_mode='r+')
            x_train_A = np.load(os.path.join(self.data_path, "cache", "{}_A.npy".format(image)), mmap_mode='r+')
            x_train = np.concatenate((x_train_M, x_train_A), axis=2)
        else:
            raise Exception("Wrong number of channels")

        y_train = np.load(os.path.join(self.data_path, "cache", "{image_id}_y.npy".format(image_id=image)))

        x_train = x_train[x:x + width, y:y + height]
        y_train = y_train[x:x + width, y:y + height]

        return x_train, y_train

    def generator(self):
        while 1:
            yield self.next(amount=1)

    def get_grid_size(self, image_number: str) -> Tuple[float, float]:
        """
        Returns the grid size of a specific image.
        Needed to scale some coords.
        """
        return self.grid_sizes[image_number]

    def read_image(self, image_id: str, band: str = 'RGB'):
        """
        Reads a image number from specified band.
        Stores the image in a numpy array.
        """
        if band == 'RGB':
            filename = os.path.join(self.data_path, 'three_band',
                                    '{}.tif'.format(image_id))
            raw_data = tifffile.imread(filename).transpose([1, 2, 0])
            image_data = scale_image_percentile(raw_data)
            return image_data
        elif band == 'M' or band == 'A':
            filename = os.path.join(self.data_path, 'sixteen_band',
                                    '{}_{}.tif'.format(image_id, band))
            raw_data = tifffile.imread(filename).transpose([1, 2, 0])
            image_data = scale_image_percentile(raw_data)
            return image_data
        else:
            raise Exception("Band not implemented")

    def scale_coords(self, img_size: Tuple[int, int], image_number: str) -> \
            Tuple[float, float]:
        """
        Get a scaling factor needed to scale polygons to same size as image
        """
        x_max, y_min = self.grid_sizes.loc[image_number][['Xmax', 'Ymin']]
        h, w = img_size
        w_ = w * (w / (w + 1))
        h_ = h * (h / (h + 1))
        return w_ / x_max, h_ / y_min

    def get_ground_truth_polys(self, image_number: str) -> \
            Dict[str, MultiPolygon]:
        """
        Get a list of polygons sorted by class for the selected image.
        Scaled to match image.
        """
        train_polygons = dict()
        for _im_id, _poly_type, _poly in csv.reader(
                open(os.path.join(self.data_path, 'train_wkt_v4.csv'))):
            if _im_id == image_number:
                train_polygons[_poly_type] = shapely.wkt.loads(_poly)

        x_scale, y_scale = self.scale_coords(
            self.read_image(image_number).shape[:2], image_number)

        train_polygons_scaled = dict()
        for key, train_polygon in train_polygons.items():
            train_polygons_scaled[key] = \
                shapely.affinity.scale(train_polygon,
                                       xfact=x_scale,
                                       yfact=y_scale,
                                       origin=(0, 0, 0))

        return train_polygons_scaled

    @staticmethod
    def get_ground_truth_array(polygons, class_number: int,
                               image_size: Tuple[int, int]):
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
                     for poly in polygons[str(class_number)] for pi in
                     poly.interiors]

        cv2.fillPoly(img_mask, interiors, 0)

        return img_mask

    def flatten(self, arr):
        # 11 class background
        img_mask = np.full((arr.shape[0], arr.shape[1]), 0, np.uint8)

        # Sort polygons by Z-order
        for cls, _ in sorted(ZORDER.items(), key=lambda x: x[1]):
            if cls-1 not in self.classes:
                continue

            mask = arr[:, :, cls-1].astype('uint8')
            mask = mask * cls

            # Create a mask to only copy pixels which are in this class
            m = np.ma.masked_where(mask > 0, mask).mask

            np.copyto(img_mask, mask, where=m)

        return img_mask

    def get_test_patches(self, image, network_size):
        print('Generating patches for image {}'.format(image))

        cache_path = os.path.join(self.data_path, "cache")

        if self.channels == 3:
            x_path = os.path.join(cache_path, "{}_x.npy".format(image))
            if os.path.isfile(x_path):
                x_train = np.load(os.path.join(cache_path, "{}_x.npy".format(image)))
            else:
                raise Exception("No data found for image {}".format(image))
        elif self.channels == 8:
            x_path = os.path.join(cache_path, "{}_A.npy".format(image))
            if os.path.isfile(x_path):
                x_train = np.load(os.path.join(cache_path, "{}_A.npy".format(image)))
            else:
                raise Exception("No data found for image {}".format(image))
        elif self.channels == 16:
            x_train_M = np.load(os.path.join(self.data_path, "cache", "{}_M.npy".format(image)), mmap_mode='r+')
            x_train_A = np.load(os.path.join(self.data_path, "cache", "{}_A.npy".format(image)), mmap_mode='r+')
            x_train = np.concatenate((x_train_M, x_train_A), axis=2)
        else:
            raise Exception("Invalid number of channels")

        y_path = os.path.join(cache_path, "{}_y.npy".format(image))
        if os.path.isfile(y_path):
            y_train = np.load(y_path)
        else:
            y_train = None
            print("No ground truth for image {}".format(image))

        image_width = x_train.shape[0]
        image_height = x_train.shape[1]

        print('Width: {} - Height: {}'.format(image_width, image_height))

        # Integer ceil division
        splits = max(-(-image_width // network_size),
                     -(-image_height // network_size))

        new_size = splits * network_size

        x_train_pad = np.zeros((new_size, new_size, self.channels))
        x_train_pad[:x_train.shape[0], :x_train.shape[1], :] = x_train

        if y_train is not None:
            y_train_pad = np.zeros((new_size, new_size, 10))
            y_train_pad[:y_train.shape[0], :y_train.shape[1], :] = y_train

        print('Splits: {}'.format(splits * splits))

        x = np.empty((splits * splits, network_size, network_size, self.channels))

        if y_train is not None:
            y = np.empty((splits * splits, network_size, network_size, 10))
        else:
            y = None

        for col in range(splits):
            for row in range(splits):
                x_start = network_size * col
                y_start = network_size * row

                x[col * splits + row] = x_train_pad[
                                        x_start:x_start + network_size,
                                        y_start:y_start + network_size]
                if y_train is not None:
                    y[col * splits + row] = y_train_pad[
                                            x_start:x_start + network_size,
                                            y_start:y_start + network_size]

        if y is not None:
            y = np.array(y)

        x = np.array(x)

        return x, y, new_size, splits, image_width, image_height
