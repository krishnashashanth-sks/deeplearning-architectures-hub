import json
import cv2
import os
import numpy as np

def load_multiview_images(image_dir):
    """
    Loads multi-view images from a specified directory.

    Args:
        image_dir (str): Path to the directory containing image files.

    Returns:
        list: A list of loaded images as NumPy arrays (RGB format).
    """
    images = []
    image_paths = []
    supported_formats = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

    if not os.path.isdir(image_dir):
        print(f"Error: Directory not found at {image_dir}")
        return []

    print(f"Loading images from: {image_dir}")
    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith(supported_formats):
            filepath = os.path.join(image_dir, filename)
            img = cv2.imread(filepath)
            if img is not None:
                # OpenCV loads images in BGR format by default, convert to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)
                image_paths.append(filepath)
            else:
                print(f"Warning: Could not load image {filepath}")

    if not images:
        print(f"No supported images found in {image_dir}")
    else:
        print(f"Successfully loaded {len(images)} images.")

    return images, image_paths

def load_camera_parameters(camera_params_path):
    """
    Loads camera intrinsic and extrinsic parameters from a JSON file.

    Args:
        camera_params_path (str): Path to the JSON file containing camera parameters.

    Returns:
        dict: A dictionary where keys are image identifiers (e.g., filenames)
              and values are dictionaries containing 'intrinsics' and 'extrinsics'.
              Intrinsics will have 'focal_length' (fx, fy) and 'principal_point' (cx, cy).
              Extrinsics will have 'rotation_matrix' (3x3) and 'translation_vector' (3x1).
    """
    camera_parameters = {}
    try:
        with open(camera_params_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Camera parameters file not found at {camera_params_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {camera_params_path}")
        return {}

    print(f"Loading camera parameters from: {camera_params_path}")
    for image_id, params in data.items():
        intrinsics = {
            "focal_length": np.array(params["focal_length"]),
            "principal_point": np.array(params["principal_point"])
        }
        extrinsics = {
            "rotation_matrix": np.array(params["rotation_matrix"]),
            "translation_vector": np.array(params["translation_vector"])
        }
        camera_parameters[image_id] = {
            "intrinsics": intrinsics,
            "extrinsics": extrinsics
        }

    if not camera_parameters:
        print(f"No camera parameters found in {camera_params_path}")
    else:
        print(f"Successfully loaded parameters for {len(camera_parameters)} cameras.")

    return camera_parameters

# Dummy load_point_cloud function (Open3D import is usually heavy)
# Replace with actual open3d logic if needed, but for dummy data, it's not strictly necessary.
# import open3d as o3d
def load_point_cloud(ply_path):
    """
    Loads a 3D point cloud from a .ply file using Open3D.
    (Dummy implementation for now)
    """
    print(f"[Dummy] Loading point cloud from: {ply_path}")
    # In a real scenario, this would use open3d.io.read_point_cloud(ply_path)
    # and extract points, colors, normals.

    # Return dummy data for demonstration
    num_points = 100
    points = np.random.rand(num_points, 3) * 10 - 5 # x, y, z coordinates
    colors = np.random.randint(0, 256, size=(num_points, 3)) # RGB colors
    normals = np.random.rand(num_points, 3) # Dummy normals
    print(f"[Dummy] Successfully loaded {len(points)} points.")
    return points, colors, normals