import os

import matplotlib.pyplot as plt
import tifffile
import numpy as np
import pandas as pd

LABEL_TO_CLASS = {
    'LARGE_BUILDING': 1,
    'RESIDENTIAL_BUILDING': 1,
    'NON_RESIDENTIAL_BUILDING': 1,
    'MISC_SMALL_STRUCTURE': 2,
    'GOOD_ROADS': 3,
    'POOR_DIRT_CART_TRACK': 4,
    'FOOTPATH_TRAIL': 4,
    'WOODLAND': 5,
    'HEDGEROWS': 5,
    'GROUP_TREES': 5,
    'STANDALONE_TREES': 5,
    'CONTOUR_PLOUGHING_CROPLAND': 6,
    'ROW_CROP': 6,
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
    """Fixes the pixel value range to 2%-98% original distribution of values"""
    orig_shape = matrix.shape
    matrix = np.reshape(matrix, [matrix.shape[0] * matrix.shape[1], 3]).astype(float)

    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins

    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, orig_shape)
    matrix = matrix.clip(0, 1)
    return matrix


class Generator:
    def __init__(self, data_path: str = "data", augment: bool = True):
        self.data_path = data_path
        self.augment = augment

        self.grid_sizes = pd.read_csv(os.path.join(self.data_path, 'grid_sizes.csv'),
                                      names=['image_id', 'x_max', 'y_min'], skiprows=1)
        self.all_image_ids = self.grid_sizes.image_id.unique()
        self.training_image_ids = [f for f in os.listdir(os.path.join(self.data_path, "train_geojson_v3"))
                                   if os.path.isdir(os.path.join(os.path.join(self.data_path, "train_geojson_v3"), f))]

    def get_grid_size(self, image_number: str):
        return self.grid_sizes[image_number]

    def read_image(self, image_number: str, band: int = 3):
        """
        Reads a image number from specified band and stores the image in a numpy array
        """
        if band == 3:
            filename = os.path.join(self.data_path, "three_band", f'{image_number}.tif')
            raw_data = tifffile.imread(filename).transpose([1, 2, 0])
            image_data = scale_image_percentile(raw_data)
            return image_data
        else:
            raise Exception("Only 3-band is implemented")

    def save_image(self, image_number: str, filename: str, overlay: bool = False, band: int = 3) -> None:
        """
        Saves a image number from specified band and saves it to filename
        Overwrites existing files without warning
        """
        image_data = self.read_image(image_number, band)

        if overlay:
            overlay_data = self.get_overlay(image_number)
            # TODO: Add overlay to image_data

        plt.imsave(filename, image_data)

    def get_overlay(self, image_number: str):
        return self.get_ground_truth(image_number)

    def save_overlay(self, image_number: str, filename: str) -> None:
        """
        Saves a image number from specified band and saves it to filename
        Overwrites existing files without warning
        """
        overlay_data = self.get_overlay(image_number)
        plt.imsave(filename, overlay_data)

    def get_ground_truth(self, image_number: str):
        folder = os.path.join(self.data_path, "train_geojson_v3", f'{image_number}')
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and not f.startswith("Grid")]
        print(files)
        return None


generator = Generator()
generator.save_image("6010_1_2", "6010_1_2.png", overlay=True)
generator.save_overlay("6010_1_2", "6010_1_2_overlay.png")
generator.get_ground_truth("6010_1_2")
