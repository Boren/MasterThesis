import os

import numpy as np
import matplotlib.pyplot as plt
import webcolors
import cv2
from PIL import Image
from tqdm import tqdm

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
    1: 'Building',
    2: 'Structure',
    3: 'Road',
    4: 'Trail',
    5: 'Trees',
    6: 'Farmland',
    7: 'Waterway',
    8: 'Still Water',
    9: 'Large Vehicle',
    10: 'Small Vehicle'
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


def mask_for_array(arr):
    # White background
    img_mask = np.full((arr.shape[0], arr.shape[1], 3), 255, np.uint8)

    # Sort polygons by Z-order
    for cls, _ in sorted(ZORDER.items(), key=lambda x: x[1]):
        mask = np.repeat(arr[:, :, cls-1, np.newaxis], 3, axis=2)
        mask = mask * webcolors.hex_to_rgb(COLOR_MAPPING[int(cls)])
        mask = mask.astype('uint8')

        # Create a mask to only copy pixels which are in this class
        m = np.ma.masked_where(mask > 0, mask).mask

        np.copyto(img_mask, mask, where=m)

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


if __name__ == "__main__":
    os.makedirs('../images/visualize_data/', exist_ok=True)

    all_files = os.listdir("../data/validation_geojson") + os.listdir("../data/train_geojson")
    all_images = [f.split(os.extsep)[0] for f in all_files if not f.startswith('.')]
    images_path = '../images/visualize_data/'

    for image in tqdm(all_images, desc='Plotting spectral bands', unit='image'):
        file_path = os.path.join(images_path, '{}_x.png'.format(image))
        if not os.path.isfile(file_path):
            # Save RGB
            x_train = np.load(os.path.join("../data/cache/{image_id}_x.npy".format(image_id=image)))

            # 0->1 to RGB
            x_train = (x_train * np.array(255)).astype(np.uint8)

            x_img = Image.fromarray(x_train)
            x_img.save(file_path)

        # Save ground truth
        file_path = os.path.join(images_path, '{}_y.png'.format(image))
        if not os.path.isfile(file_path):
            y_train = np.load(os.path.join("../data/cache/{image_id}_y.npy".format(image_id=image)))

            y_mask = mask_for_array(y_train)
            y_img = Image.fromarray(y_mask)
            y_img.save(file_path)

        # Save 16-band
        x_train_M = np.load(os.path.join("../data/cache/{image_id}_M.npy".format(image_id=image)))

        for band in range(x_train_M.shape[2]):
            file_path = os.path.join(images_path, '{}_M_{}.png'.format(image, band))
            if not os.path.isfile(file_path):
                band_data = x_train_M[:, :, band]
                band_data *= 255.0 / band_data.max()

                cm = plt.get_cmap('Spectral')
                colored_image = cm(band_data.astype(np.uint8))

                x_img = Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))
                x_img.save(file_path)

        # Save IR band
        x_train_A = np.load(os.path.join("../data/cache/{image_id}_A.npy".format(image_id=image)))

        for band in range(x_train_A.shape[2]):
            file_path = os.path.join(images_path, '{}_A_{}.png'.format(image, band))
            if not os.path.isfile(file_path):
                band_data = x_train_A[:, :, band]
                band_data *= 255.0 / band_data.max()

                cm = plt.get_cmap('Spectral')
                colored_image = cm(band_data.astype(np.uint8))

                x_img = Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))
                x_img.save(file_path)

    # Make bar graph
    pixel_count = []
    labels = []

    for image in tqdm(all_images, desc='Calculating class distributions', unit='image'):
        y_train = np.load(os.path.join(
            "../data/cache/{image_id}_y.npy".format(image_id=image)))

        image_count = np.zeros(10, dtype=int)
        for cls in range(10):
            count = np.count_nonzero(y_train[:, :, cls])

            image_count[cls] = count

        pixel_count.append(image_count)
        labels.append(image)

    totals = np.sum(pixel_count, axis=0)
    total_pixels = np.sum(pixel_count)
    totals = (totals / total_pixels) * 100

    classes = []
    colors = []

    for cls in range(10):
        classes.append(CLASS_TO_LABEL[cls+1])
        colors.append(COLOR_MAPPING[int(cls+1)])

    # this is for plotting purpose
    index = np.arange(10)
    plt.bar(index, totals, color=colors)
    #plt.xlabel('Class')
    plt.ylabel('% of total')
    plt.xticks(index, classes, rotation=30)
    #plt.title('Total Class Distribution')
    plt.show()
