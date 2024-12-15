# 3D Gaussian Splatting Manipulation Project

This repository contains our approach to manipulate and visualize 3D gaussian splats, with scene
navigation and gaussians labeling functionalities.

## Overview

The codebase has 4 main components:

- `gaussians_selection.js` that contains the implementation of the viewer with all gaussians manipulation
  functionalities
- `index.html` that initializes the page
- `deep_learning_segmentation.py` that assigns labels to gaussians by applying our "Majority Vote" mechanism on
  segmented images
- `3D_clustering/k_means.py` that assigns labels to gaussians using our KMEANS clustering algorithm

## Software Requirements

- Conda (recommended for easy setup)
- Requirements are specified in `environment.yml`
- WebGL 1.0 or later

## Running

### Starting a Web Server

You can start a simple web server in the relevant directory:

Using Python:

```shell
cd your_directory/; python3 -m http.server [portNumber]
```

Then access it from http://localhost:portNumber/

Alternatively, you can use Live Server if you edit your code with Visual Studio Code:
https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer

### Loading a Labelled PLY File

If you already have a PLY file labeled by one of our labelisation programs (`deep_learning_segmentation.py` or
`k_means.py`), you can simply drag and drop that file into viewer's page.

### Gaussians Labelisation

To assign labels to gaussians you can use either of our labelisation programs:

### Using Deep Learning Segmentation

To run `deep_learning_segmentation.py`, you need the following files:

- ply_file: Input file with gaussians PLY file
- camera_file: JSON file with camera data of the scene you'd like to generate labels for
- input_dir: Directory containing input images of the scene

Then run (set the `model` parameter to either `mask2former`, `segformer`, or `yolo`):

```shell
python deep_learning_segmentation.py \
    --ply_file "path/to/point_cloud.ply" \
    --camera_file "path/to/cameras.json" \
    --input_dir "path/to/input/images" \
    --output_dir "path/to/output/dir" \
    --output_file "path/to/output/file.ply" \
    --model mask2former
```

### Using K-means Clustering

For `k_means.py`, you only need the input PLY file. Run:

```shell
python kmeans.py --file_path "path/to/point_cloud.ply" --save_path "path/to/save/file.ply"
```

## Controls

### Gaussian Controls

- ESC - toggle selection mode<br>
- Click - select object<br>
- H/K - move left/right<br>
- U/J - move up/down<br>
- Del - delete an object<br>
- R - reset all movements<br>
- Shift+R - reset all objects

### Movement (Arrow Keys)

- Left/right arrow keys to strafe side to side
- Up/down arrow keys to move forward/back
- `space` to jump

### Camera Angle (WASD)

- `a`/`d` to turn camera left/right
- `w`/`s` to tilt camera up/down
- `q`/`e` to roll camera counterclockwise/clockwise
- `i`/`k` and `j`/`l` to orbit

### Trackpad

- Scroll up/down to orbit down
- Scroll left/right to orbit left/right
- Pinch to move forward/back
- Ctrl key + scroll up/down to move forward/back
- Shift + scroll up/down to move up/down
- Shift + scroll left/right to strafe side to side

### Mouse

- Click and drag to orbit
- Right click (or ctrl/cmd key) and drag up/down to move forward/back
- Right click (or ctrl/cmd key) and drag left/right to strafe side to side

### Touch (Mobile)

- One finger to orbit
- Two finger pinch to move forward/back
- Two finger rotate to rotate camera clockwise/counterclockwise
- Two finger pan to move side-to-side and up-down

### Other

- Press 0-9 to switch to one of the pre-loaded camera views
- Press '-' or '+' key to cycle loaded cameras
- Press `p` to resume default animation
- Drag and drop .ply file to convert to .splat
- Drag and drop cameras.json to load cameras
  I'll add an "Authors" section at the end of the README, following standard practices for academic/project
  documentation.

## Authors

- **Gloire LINVANI** - *École Polytechnique, Télécom Paris, France*
- **Onur BASCI**
- **Dimah BASSAL**

## Acknowledgments

We cloned the base version of the viewer from: https://github.com/antimatter15/splat.git