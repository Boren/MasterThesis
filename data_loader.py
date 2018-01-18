import csv
import os
import sys

import cv2
import matplotlib.pyplot as plt
import tifffile
import numpy as np
import pandas as pd
import webcolors

import shapely.wkt
import shapely.affinity

csv.field_size_limit(sys.maxsize)

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
    def __init__(self, data_path: str = "data", augment: bool = True):
        self.data_path = data_path
        self.augment = augment

        # TODO: Pre-fetch image sizes
        self.grid_sizes = pd.read_csv(os.path.join(self.data_path, 'grid_sizes.csv'), index_col=0)
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

    def save_image(self, image_number: str, filename: str, band: int = 3) -> None:
        """
        Saves a image number from specified band and saves it to filename
        Overwrites existing files without warning
        """
        image_data = self.read_image(image_number, band)
        plt.imsave(filename, image_data)

    def save_overlay(self, image_number: str, filename: str) -> None:
        """
        Saves a image number from specified band and saves it to filename
        Overwrites existing files without warning
        """
        overlay_data = self.get_ground_truth_polys(image_number)
        train_mask = self.mask_for_polygons(overlay_data, image_number)
        plt.imsave(filename, train_mask)

    def scale_coords(self, img_size, image_number: str):
        """Scale the coordinates of a polygon into the image coordinates for a grid cell"""
        x_max, y_min = self.grid_sizes.loc[image_number][['Xmax', 'Ymin']]
        h, w = img_size
        w_ = w * (w / (w + 1))
        h_ = h * (h / (h + 1))
        return w_ / x_max, h_ / y_min

    def get_ground_truth_polys(self, image_number: str):
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

    def mask_for_polygons(self, polygons, image_number: str):
        """
        Create a color mask of classes with the same size as original image
        """
        w, h = self.read_image(image_number).shape[:2]

        img_mask = np.full((w, h, 3), 255, np.uint8)

        # Sort polygons by Z-order
        for cls, _ in sorted(ZORDER.items(), key=lambda x: x[1]):
            exteriors = [np.array(poly.exterior.coords.round().astype(np.int32)) for poly in polygons[str(cls)]]
            cv2.fillPoly(img_mask, exteriors, webcolors.hex_to_rgb(COLOR_MAPPING[int(cls)]))

        return img_mask


if __name__ == "__main__":
    generator = Generator()

    for training_image in os.listdir("data/train_geojson_v3"):
        generator.save_image(training_image, f"images/{training_image}.png")
        generator.save_overlay(training_image, f"images/{training_image}_overlay.png")
