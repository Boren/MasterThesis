import os
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.collections import PatchCollection
from matplotlib.transforms import Bbox

DATA_DIRECTORY = "data"
IMAGE_DIRECTORY = "three_band"
IMAGE = "6070_2_3"

# Load grid CSV
grid_sizes = pd.read_csv(os.path.join(DATA_DIRECTORY, 'grid_sizes.csv'), index_col=0)

im_fname = os.path.join(DATA_DIRECTORY, IMAGE_DIRECTORY, f'{IMAGE}.tif')
tif_data = tifffile.imread(im_fname).transpose([1,2,0])

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


def scale_coords(tif_data, grid_name, point):
    """Scale the coordinates of a polygon into the image coordinates for a grid cell"""
    w, h, _ = tif_data.shape
    Xmax, Ymin = grid_sizes.loc[grid_name][['Xmax', 'Ymin']]
    x, y = point[:, 0], point[:, 1]

    wp = float(w ** 2) / (w + 1)
    xp = x / Xmax * wp

    hp = float(h ** 2) / (h + 1)
    yp = y / Ymin * hp

    return np.concatenate([xp[:, None], yp[:, None]], axis=1)


def scale_percentile(matrix):
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

fixed_im = scale_percentile(tif_data)
plt.imsave("test.png", fixed_im)


def load_overlays(tile_name):
    """Get all of the polygon overlays for a tile.
    Returns a dict: {LABEL: POLYGON}"""
    dirname = os.path.join(DATA_DIRECTORY, 'train_geojson_v3/%s/' % tile_name)
    fnames = [os.path.join(dirname, fname) for fname in
              os.listdir(dirname)
              if fname.endswith('.geojson') and not fname.startswith('Grid')]

    overlays = dict()
    for fname in fnames:
        with open(fname, 'r') as f:
            sh_json = json.load(f)
        label = sh_json['features'][0]['properties']['LABEL']
        superclass = sh_json['features'][0]['properties']['SUPERCLASS']
        print(f"{label} ({superclass})")

        polygons = []
        for sh in sh_json['features']:
            pts = scale_coords(tif_data, IMAGE, np.array(sh['geometry']['coordinates'][0])).squeeze()

            # Remove badly formatted polygons
            if not ((len(pts.shape) == 2) and (pts.shape[1] == 2) and (pts.shape[0] > 2)):
                continue
            polygons.append(pts)

        overlays[label] = polygons

    return overlays


overlays_poly = load_overlays(IMAGE)

fig, ax = plt.subplots(figsize=(36, 36))
#ax.imshow(fixed_im)

for label in LABEL_TO_CLASS:
    if label in overlays_poly:
        patches = []
        for pts in overlays_poly[label]:
            poly = matplotlib.patches.Polygon(pts)
            patches.append(poly)

        facecolor = COLOR_MAPPING[LABEL_TO_CLASS[label]]
        edgecolor = [int(facecolor[i:i+2], 16) for i in (1, 3 ,5)] + [255]
        edgecolor = np.array(edgecolor) / 255.0
        edgecolor *= 0.5

        p = PatchCollection(patches,
                            facecolors=facecolor,
                            edgecolors=edgecolor,
                            alpha=1.0)

        ax.add_collection(p)

w, h, _ = tif_data.shape

ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])
plt.axis('off')
plt.xlim(0, h)
plt.ylim(w, 0)

fig.savefig("test_overlay.png", transparent=False, bbox_inches='tight', pad_inches=0)
