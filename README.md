# Deep Convolutional Neural Networks for Semantic Segmentation of Multi-Band Satellite Images

<img src="repository/6070_2_3_y.png?raw=true" width="400px"> <img src="repository/6070_2_3_x.png?raw=true" width="400px">

## Preparation

- Download 3-band and 16-band from [here](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data) and extract to data folders
- Install requirements by executing:

` $ pip install -r requirements.txt `
- In addition you need to install either _tensorflow_ or _tensorflow-gpu_

## Training

`$ python train.py`


| Argument      | Description            | Options                                                  |
| ------------- | ---------------------- | -------------------------------------------------------- |
| `--algorithm` | Algorithm to train     | `unet`, `fcn_densenet`, `tiramisu`, `pspnet`             |
| `--size`      | Size of patches        | _int_                                                    |
| `--epochs`    | Epochs to train for    | _int_                                                    |
| `--batch`     | Samples per batch      | _int_                                                    |
| `--channels`  | Image channels         | `3`, `8`, `16`                                           |
| `--loss`      | Loss function          | `crossentropy`, `jaccard`, `dice`, `cejaccard`, `cedice` |
| `--verbose`   | Print more information | _bool_                                                   |
| `--noaugment` | Turn off augmentation  | _bool_                                                   |
| `--name`      | Give run a custom name | str                                                      |


## Testing

`$ python train.py --test`


| Argument      | Description            | Options                                                  |
| ------------- | ---------------------- | -------------------------------------------------------- |
| `--algorithm` | Algorithm to test      | `unet`, `fcn_densenet`, `tiramisu`, `pspnet`             |
| `--size`      | Size of patches        | _int_                                                    |
| `--channels`  | Image channels         | `3`, `8`, `16`                                           |
| `--verbose`   | Print more information | _bool_                                                   |

## Visualization

It's possible to run some visualization of the data by running `$ python visualize.py` from the _utils_ folder.

<img src="repository/data_distribution.png?raw=true" width="600px">

