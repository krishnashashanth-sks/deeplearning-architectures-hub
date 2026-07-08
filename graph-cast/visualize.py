import matplotlib.pyplot as plt

# ---  Visualization Function ---

def visualize_graph(data, output_inference=None, feature_idx=0, title="Graph Node Visualization"):
    """Visualizes the graph nodes and optionally inference output."""
    print("\nGenerating graph visualization...")
    plt.figure(figsize=(12, 8))

    # Convert edge_index to NetworkX graph for plotting edges if needed
    # Note: For very large graphs, plotting all edges can be slow and cluttered.
    # For global weather data, a scatter plot of nodes is often more informative.
    
    # Scatter plot of nodes (latitude, longitude)
    lats = data.pos[:, 0].numpy()
    lons = data.pos[:, 1].numpy()

    if output_inference is not None:
        # Visualize the first feature from the output
        feature_values = output_inference[:, feature_idx].numpy()
        scatter = plt.scatter(lons, lats, c=feature_values, cmap='viridis', s=10, alpha=0.7)
        plt.colorbar(scatter, label=f'Output Feature {feature_idx} Value')
        plt.title(f'{title} (Output Feature {feature_idx})')
    else:
        # Just visualize nodes without specific feature values
        plt.scatter(lons, lats, s=5, alpha=0.5, color='blue')
        plt.title(f'{title} (Nodes Only)')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()