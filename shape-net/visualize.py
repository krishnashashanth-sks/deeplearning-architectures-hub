import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

def visualize_point_cloud(point_cloud, title="Reconstructed Point Cloud"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Convert TensorFlow tensor to NumPy array for plotting
    if tf.is_tensor(point_cloud):
        point_cloud_np = point_cloud.numpy()
    else:
        point_cloud_np = point_cloud

    # Extract x, y, z coordinates
    x = point_cloud_np[:, 0]
    y = point_cloud_np[:, 1]
    z = point_cloud_np[:, 2]

    # Plot points
    ax.scatter(x, y, z, s=1, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()