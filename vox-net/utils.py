import trimesh
import numpy as np
from skimage.measure import marching_cubes

def mesh_to_voxel_grid(off_filepath, resolution=32, num_points=2048):
    """
    Loads an OFF mesh, samples points, and converts them to a binary voxel grid.
    """
    try:
        # Load the mesh using trimesh
        mesh = trimesh.load(off_filepath)

        # Sample points from the mesh surface
        if isinstance(mesh, trimesh.Trimesh):
            points, face_indices = mesh.sample(num_points, return_index=True)
        elif isinstance(mesh, trimesh.PointCloud):
            points = mesh.vertices # If it's already a point cloud
            if len(points) > num_points:
                # Randomly sample if more points than desired
                idx = np.random.choice(len(points), num_points, replace=False)
                points = points[idx]
            elif len(points) < num_points:
                # Pad with duplicates if fewer points than desired (simple approach)
                idx = np.random.choice(len(points), num_points, replace=True)
                points = points[idx]
        else:
            print(f"Warning: Unsupported mesh type for {off_filepath}. Skipping.")
            return None

        if len(points) == 0:
            print(f"Warning: No points sampled for {off_filepath}. Skipping.")
            return None

        # Normalize points to fit within a [0, 1] bounding box
        points = points - points.min(axis=0) # Translate to non-negative
        points = points / points.max(axis=0) # Scale to [0, 1]

        # Convert normalized points to voxel coordinates
        voxel_coords = np.floor(points * (resolution - 1)).astype(int)

        # Create an empty voxel grid
        voxel_grid = np.zeros((resolution, resolution, resolution), dtype=np.bool_)

        # Mark occupied voxels
        for coord in voxel_coords:
            if all(0 <= c < resolution for c in coord):
                voxel_grid[coord[0], coord[1], coord[2]] = 1

        return voxel_grid

    except Exception as e:
        print(f"Error processing {off_filepath}: {e}")
        return None

def mesh_to_voxel_grid(off_filepath, resolution=32, num_points=2048):
    """
    Loads an OFF mesh, samples points, and converts them to a binary voxel grid.
    """
    try:
        # Load the mesh using trimesh
        mesh = trimesh.load(off_filepath)

        # Sample points from the mesh surface
        if isinstance(mesh, trimesh.Trimesh):
            points = mesh.sample(num_points) # trimesh.sample directly returns points
        elif isinstance(mesh, trimesh.PointCloud):
            points = mesh.vertices # If it's already a point cloud
            if len(points) > num_points:
                idx = np.random.choice(len(points), num_points, replace=False)
                points = points[idx]
            elif len(points) < num_points:
                # Pad with duplicates if fewer points than desired (simple approach)
                idx = np.random.choice(len(points), num_points, replace=True)
                points = points[idx]
        else:
            # Handle other trimesh types or unexpected cases
            # For example, if it's a scene with multiple meshes, convert to a single mesh first
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) > 0:
                    # Try to combine into a single mesh, or pick the largest
                    points = np.vstack([g.sample(num_points // len(mesh.geometry) + 1) for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
                    if len(points) > num_points:
                        idx = np.random.choice(len(points), num_points, replace=False)
                        points = points[idx]
                    elif len(points) < num_points:
                        idx = np.random.choice(len(points), num_points, replace=True)
                        points = points[idx]
                else:
                    print(f"Warning: Scene {off_filepath} has no geometry. Skipping.")
                    return None
            else:
                print(f"Warning: Unsupported mesh type {type(mesh)} for {off_filepath}. Skipping.")
                return None

        if len(points) == 0:
            print(f"Warning: No points sampled for {off_filepath}. Skipping.")
            return None

        # Normalize points to fit within a [0, 1] bounding box
        # Ensure non-zero extent to avoid division by zero for flat shapes
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        extent = max_coords - min_coords
        # Add a small epsilon to avoid division by zero if extent is 0 in some dimension
        extent[extent == 0] = 1e-6

        points = (points - min_coords) / extent

        # Convert normalized points to voxel coordinates
        voxel_coords = np.floor(points * (resolution - 1)).astype(int)

        # Create an empty voxel grid
        voxel_grid = np.zeros((resolution, resolution, resolution), dtype=np.bool_)

        # Mark occupied voxels
        for coord in voxel_coords:
            # Ensure coordinates are within bounds
            if all(0 <= c < resolution for c in coord):
                voxel_grid[coord[0], coord[1], coord[2]] = 1

        return voxel_grid

    except Exception as e:
        print(f"Error processing {off_filepath}: {e}")
        return None
