import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


def load_cameras(camera_file):
    """
    Loading camera data from JSON file
    """
    with open(camera_file, 'r') as f:
        return json.load(f)


def load_gaussians(ply_file):
    """
    Loading gaussian data from PLY file
    """
    plydata = PlyData.read(ply_file)
    vertices = plydata['vertex']

    # Creating structured array for gaussians
    gaussians = np.zeros(len(vertices),
                         dtype=[('position', np.float32, 3), ('scale', np.float32, 3), ('rotation', np.float32, 4)])

    gaussians['position'][:, 0] = vertices['x']
    gaussians['position'][:, 1] = vertices['y']
    gaussians['position'][:, 2] = vertices['z']

    return gaussians, plydata


def project_gaussian(position, camera):
    """
    Projecting a gaussian position to image space using camera parameters

    The transformation from world to camera coordinates is:
    [R | t] where t = -R @ position

    In homogeneous coordinates this would be:
    [R  | t]
    [0  | 1]
    """
    fx = camera["fx"]
    fy = camera["fy"]
    width = camera["width"]
    height = camera["height"]

    # rotation matrix (3x3)
    R = np.array(camera["rotation"])

    # camera position
    p = np.array(camera["position"])

    # translation vector (3x1)
    t = -R @ p

    # Transforming point from world to camera coordinates
    pos_cam = R @ position + t

    # projecting only if the point is in front of the camera
    if pos_cam[2] <= 0:
        return None

    # Projecting to image coordinates
    x = (fx * pos_cam[0] / pos_cam[2]) + width / 2
    y = (fy * pos_cam[1] / pos_cam[2]) + height / 2

    # Checking if the point is within image bounds
    if 0 <= x < width and 0 <= y < height:
        return (int(x), int(y))
    return None


def segment_image(image_path, output_dir, processor, model, device):
    """
    Performing semantic segmentation on a single image
    """
    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(image_path)

    # print("Original image size:", image.size)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        seg_map = outputs.logits.argmax(dim=1)[0].cpu().numpy()

    # print("Segmentation map size:", seg_map.shape)

    # Saving segmentation map
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    np.save(os.path.join(output_dir, f"{base_filename}_segmap.npy"), seg_map)

    plt.figure(figsize=(10, 10))
    plt.imshow(seg_map)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"{base_filename}_segmap.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

    return seg_map


def assign_semantic_labels(gaussians, cameras, input_dir, output_dir):
    """
    Assigning semantic labels to gaussians based on their projections in segmented images
    """
    # Setting up segmentation model
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
    model.to(device)

    # Dictionary to store votes for each gaussian
    gaussian_votes = {}

    # Processing each camera view
    for camera in cameras:
        img_path = os.path.join(input_dir, camera["img_name"])
        if not os.path.exists(img_path):
            print(f"Warning: Image {camera['img_name']} not found")
            continue
        image = Image.open(img_path)

        # Getting original image size
        orig_width, orig_height = image.size[0], image.size[1]

        # Getting segmentation map for this view
        seg_map = segment_image(img_path, output_dir, processor, model, device)
        seg_width, seg_height = seg_map.shape[1], seg_map.shape[0]

        # Computing scaling factors
        height_scale = seg_height / orig_height
        width_scale = seg_width / orig_width

        # Processing each gaussian
        for idx, gaussian in enumerate(gaussians):
            # Projecting gaussian to this camera view
            proj_pos = project_gaussian(gaussian['position'], camera)

            if proj_pos is not None:
                x, y = proj_pos

                x_scaled = int(x * width_scale)
                y_scaled = int(y * height_scale)

                # Ensuring coordinates are within bounds
                x_scaled = min(max(0, x_scaled), seg_width - 1)
                y_scaled = min(max(0, y_scaled), seg_height - 1)

                label = seg_map[y_scaled, x_scaled]

                if idx not in gaussian_votes:
                    gaussian_votes[idx] = {}

                if label not in gaussian_votes[idx]:
                    gaussian_votes[idx][label] = 0
                gaussian_votes[idx][label] += 1

    N = len(gaussians)
    # Assigning final labels based on majority voting
    labels = np.zeros(N, dtype=np.int32)
    for idx in range(N):
        if idx in gaussian_votes:
            # Getting label with most votes
            labels[idx] = max(gaussian_votes[idx].items(), key=lambda x: x[1])[0]
        else:
            # Default label for gaussians not visible in any view
            labels[idx] = -1

    return labels


def save_labeled_ply(output_file, plydata, labels):
    """
    Saving gaussians with semantic labels to new PLY file
    """
    vertices = plydata['vertex']

    # Creating new vertex type with semantic label
    vertex_dtype = vertices.data.dtype.descr + [('semantic_label', 'i4')]

    # Creating new vertex data
    new_vertices = np.empty(len(vertices), dtype=vertex_dtype)

    # Copying existing data
    for name in vertices.data.dtype.names:
        new_vertices[name] = vertices[name]

    # Adding semantic labels
    new_vertices['semantic_label'] = labels

    # Creating new PLY element and file
    vertex_element = PlyElement.describe(new_vertices, 'vertex')
    PlyData([vertex_element], text=True).write(output_file)


def main():
    parser = argparse.ArgumentParser(description='Add semantic labels to gaussian PLY file')
    parser.add_argument('ply_file', help='Input PLY file with gaussian data')
    parser.add_argument('camera_file', help='JSON file with camera data')
    parser.add_argument('input_dir', help='Directory containing input images')
    parser.add_argument('output_dir', help='Output directory to saved segmented input images')
    parser.add_argument('output_file', help='Output PLY file with semantic labels')
    args = parser.parse_args()

    # Loading data
    print("Loading cameras...")
    cameras = load_cameras(args.camera_file)

    print("Loading gaussians...")
    gaussians, plydata = load_gaussians(args.ply_file)

    # Assigning semantic labels
    print("Assigning semantic labels...")
    labels = assign_semantic_labels(gaussians, cameras, args.input_dir, args.output_dir)

    # Saving results
    print("Saving labeled PLY file...")
    save_labeled_ply(args.output_file, plydata, labels)

    print(f"Done! Labeled PLY file saved as {args.output_file}")

    # Printing some statistics
    unique_labels = np.unique(labels)
    print("\nLabel statistics:")
    print(f"Total gaussians: {len(labels)}")
    print(f"Number of unique labels: {len(unique_labels)}")
    print(f"Label counts:")
    for label in unique_labels:
        count = np.sum(labels == label)
        print(f"Label {label}: {count} gaussians ({100 * count / len(labels):.2f}%)")


if __name__ == "__main__":
    main()
