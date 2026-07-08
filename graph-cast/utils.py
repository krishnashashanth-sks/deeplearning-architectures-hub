import os
import cdsapi
import xarray as xr
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

def configure_cds_api(cds_api_key):
    """Configures the CDS API key in the .cdsapirc file."""
    cdsapi_rc_path = os.path.expanduser('~/.cdsapirc')
    try:
        with open(cdsapi_rc_path, 'w') as f:
            f.write('url: https://cds.climate.copernicus.eu/api\n')
            f.write(f'key: {cds_api_key}\n')
        print(f"CDS API configuration written to {cdsapi_rc_path}")
    except Exception as e:
        print(f"Error writing .cdsapirc file: {e}")
        raise

def download_era5_data(output_dir='data', output_filename='era5_reanalysis_sample.nc'):
    """Downloads ERA5 reanalysis data using cdsapi."""
    c = cdsapi.Client()
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_filename)

    print(f"Downloading ERA5 data to {output_file}...")
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    '2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind',
                    'surface_pressure', 'mean_sea_level_pressure',
                ],
                'date': '2021-01-01/2021-01-02',
                'time': [
                    '00:00', '06:00', '12:00', '18:00',
                ],
                'area': [
                    90, -180, -90, 180,
                ],
            },
            output_file
        )
        print('ERA5 data downloaded successfully.')
        return output_file
    except Exception as e:
        print(f"Error downloading ERA5 data: {e}")
        print("Please ensure your CDS API key is correctly set up.")
        raise

def load_era5_data(file_path):
    """Loads ERA5 data into an xarray Dataset."""
    print(f"Loading data from {file_path}...")
    try:
        ds = xr.open_dataset(file_path)
        print("ERA5 data loaded successfully into xarray Dataset.")
        return ds
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise
    except Exception as e:
        print(f"Error loading ERA5 data: {e}")
        raise

def extract_graph_features(ds):
    """Extracts node features and coordinates from the xarray Dataset."""
    input_variables = [
        't2m', 'u10', 'v10',
        'sp', 'msl'
    ]
    example_time_step = ds.isel(time=0)

    processed_variables = []
    for var_name in input_variables:
        if var_name in example_time_step:
            var_data = example_time_step[var_name].values
            processed_variables.append(var_data.flatten())
        else:
            print(f"Warning: Variable '{var_name}' not found in dataset. Skipping.")

    if not processed_variables:
        raise ValueError("No input variables found in the dataset.")

    node_features_np = np.stack(processed_variables, axis=-1)
    node_features = torch.tensor(node_features_np, dtype=torch.float32)

    latitudes = example_time_step['latitude'].values
    longitudes = example_time_step['longitude'].values
    node_coords_np = np.array([
        (lat, lon) for lat in latitudes for lon in longitudes
    ])
    node_coords = torch.tensor(node_coords_np, dtype=torch.float32)

    print(f"Shape of initial node features: {node_features.shape}")
    print(f"Shape of node coordinates: {node_coords.shape}")
    return node_features, node_coords

def generate_pyg_graph(node_features, node_coords, k_neighbors=8):
    """Generates a PyTorch Geometric Data object using k-nearest neighbors."""
    print(f"Generating k-nearest neighbors graph with k={k_neighbors}...")
    edge_index = knn_graph(node_coords, k=k_neighbors, loop=False)
    data = Data(x=node_features, edge_index=edge_index, pos=node_coords)
    print(f"Generated PyTorch Geometric Data object:")
    return data