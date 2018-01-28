import numpy as np
import matplotlib.pyplot as plt
import webcolors
import cv2

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
    2: 6,
    3: 4,
    4: 1,
    5: 3,
    6: 2,
    7: 7,
    8: 8,
    9: 9,
    10: 10
}


def mask_for_polygons(polygons, width, height):
    """
    Create a color mask of classes with the same size as original image
    """
    # White background
    img_mask = np.full((width, height, 3), 255, np.uint8)

    # Sort polygons by Z-order
    for cls, _ in sorted(ZORDER.items(), key=lambda x: x[1]):
        exteriors = [np.array(poly.exterior.coords).round().astype(np.int32) for poly in polygons[str(cls)]]
        cv2.fillPoly(img_mask, exteriors, webcolors.hex_to_rgb(COLOR_MAPPING[int(cls)]))

        # Some polygons have regions inside them which need to be excluded
        interiors = [np.array(pi.coords).round().astype(np.int32)
                     for poly in polygons[str(cls)] for pi in poly.interiors]
        cv2.fillPoly(img_mask, interiors, (255, 255, 255))

    return img_mask


def save_image_array(arr, filename: str) -> None:
    """
    Saves a image array to a file
    Overwrites existing files without warning
    """
    plt.imsave(filename, arr)


def save_overlay_polygons(polygons, width, height, filename: str) -> None:
    """
    Saves a image number from specified band and saves it to filename
    Overwrites existing files without warning
    """
    train_mask = mask_for_polygons(polygons, width, height)
    plt.imsave(filename, train_mask)
