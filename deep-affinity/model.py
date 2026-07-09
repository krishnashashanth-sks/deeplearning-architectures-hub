import torch
import torch.nn as nn
import torch.optim as optim

class DeepAffinityModel(nn.Module):
    def __init__(self, rdk_dim, maccs_dim, ecfp_dim, protein_max_len, num_amino_acids):
        super(DeepAffinityModel, self).__init__()

        # --- 1. Define Compound Branches ---

        # RDKit Descriptors Branch
        self.rdk_branch = nn.Sequential(
            nn.Linear(rdk_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # MACCS Keys Branch
        self.maccs_branch = nn.Sequential(
            nn.Linear(maccs_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # ECFP Fingerprints Branch
        self.ecfp_branch = nn.Sequential(
            nn.Linear(ecfp_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # --- 2. Define Protein Branch ---

        # Protein one-hot encoding input: (batch_size, protein_max_len, num_amino_acids)
        # For Conv1d, input should be (batch_size, channels, sequence_length)
        # So we treat num_amino_acids as channels and protein_max_len as sequence_length
        self.protein_branch = nn.Sequential(
            nn.Conv1d(in_channels=num_amino_acids, out_channels=128, kernel_size=8),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1) # Pools across the sequence dimension to a fixed size of 1
        )
        # Calculate the output dimension of the protein branch after pooling
        # The output of AdaptiveMaxPool1d will be (batch_size, out_channels, 1)
        # So, after flattening, it will be out_channels (256 in this case)
        protein_branch_output_dim = 256

        # --- 3. Define Combination/Fusion Layer ---

        # Total concatenated features from all branches
        # RDKit (256) + MACCS (128) + ECFP (512) + Protein (256) = 1152
        combined_feature_dim = 256 + 128 + 512 + protein_branch_output_dim

        self.combination_layer = nn.Sequential(
            nn.Linear(combined_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # --- 4. Define Prediction Head ---
        self.prediction_head = nn.Linear(256, 1) # Output a single affinity value

    def forward(self, rdk_features, maccs_features, ecfp_features, protein_features):
        # Compound feature processing
        rdk_out = self.rdk_branch(rdk_features.float())
        maccs_out = self.maccs_branch(maccs_features.float())
        ecfp_out = self.ecfp_branch(ecfp_features.float())

        # Protein feature processing
        # Reshape for Conv1d: (batch_size, max_len, num_amino_acids) -> (batch_size, num_amino_acids, max_len)
        protein_features_reshaped = protein_features.permute(0, 2, 1).float()
        protein_out = self.protein_branch(protein_features_reshaped)
        protein_out = protein_out.view(protein_out.size(0), -1) # Flatten the output of AdaptiveMaxPool1d

        # Concatenate all features
        combined_features = torch.cat((rdk_out, maccs_out, ecfp_out, protein_out), dim=1)

        # Pass through combination layer
        combined_out = self.combination_layer(combined_features)

        # Prediction
        affinity_prediction = self.prediction_head(combined_out)
        return affinity_prediction
