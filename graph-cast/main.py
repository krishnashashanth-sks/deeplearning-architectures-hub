# Ensure necessary libraries are installed
# pip install --quiet "cdsapi>=0.7.7" xarray netCDF4 matplotlib networkx
# pip install --quiet torch_geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-${{torch.__version__}}.html
from torch_geometric.loader import DataLoader
from utils import configure_cds_api,download_era5_data,load_era5_data,extract_graph_features,generate_pyg_graph
from model import GraphCastModel
from inference import run_inference
from visualize import visualize_graph

cds_api_key = "065fcd49-3d0c-4b55-bc46-77dd0b67c64c" # Example key

try:
    configure_cds_api(cds_api_key)
    era5_file_path = download_era5_data()
    era5_dataset = load_era5_data(era5_file_path)

    node_features, node_coords = extract_graph_features(era5_dataset)
    graph_data = generate_pyg_graph(node_features, node_coords)

    # Define model dimensions
    input_node_features_dim = node_features.shape[1]
    node_coords_dim = node_coords.shape[1]
    batch_size=1
    latent_dim=128
    edge_dim=0

    output_dim = input_node_features_dim # For auinput_node_features_dim-like setup
    graphcast_model = GraphCastModel(
        input_dim=input_node_features_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        coord_dim=node_coords_dim,
        num_message_passing_steps=18,
        edge_dim=edge_dim
    )

    single_graph_dataset = [graph_data]
    data_loader = DataLoader(
        single_graph_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    print(f"DataLoader created with batch_size={batch_size}.")

    output_inference, input_batch = run_inference(graphcast_model, data_loader)

    if output_inference is not None:
        # Visualize the first output feature (e.g., predicted 2m_temperature)
        visualize_graph(input_batch, output_inference, feature_idx=0, title="Inference Output: Predicted 2m Temperature")
        # You can also visualize an input feature
        visualize_graph(input_batch, input_batch.x, feature_idx=0, title="Input Feature: Actual 2m Temperature")

except Exception as e:
    print(f"An error occurred during the overall execution: {e}")
