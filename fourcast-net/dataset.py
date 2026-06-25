from torch.utils.data import Dataset

class SpatioTemporalDataset(Dataset):
    def __init__(self, data, input_sequence_length, output_sequence_length):
        self.data = data # (num_samples, num_total_time_steps, channels, height, width)
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length

        # Determine how many (input, target) pairs can be formed from each sample
        self.max_start_idx = data.shape[1] - (input_sequence_length + output_sequence_length) + 1
        if self.max_start_idx <= 0:
            raise ValueError(
                "Not enough time steps in data to form even one input-target pair."+
                f"Requires at least {input_sequence_length + output_sequence_length} time steps, got {data.shape[1]}."
            )

    def __len__(self):
        # Total number of possible (input, target) pairs across all samples
        return self.data.shape[0] * self.max_start_idx

    def __getitem__(self, idx):
        sample_idx = idx // self.max_start_idx
        start_time_step = idx % self.max_start_idx

        # Input sequence: [start_time_step, ..., start_time_step + input_sequence_length - 1]
        input_sequence = self.data[sample_idx, start_time_step : start_time_step + self.input_sequence_length]

        # Target sequence: [start_time_step + input_sequence_length, ..., start_time_step + input_sequence_length + output_sequence_length - 1]
        target_sequence = self.data[sample_idx, start_time_step + self.input_sequence_length : start_time_step + self.input_sequence_length + self.output_sequence_length]

        # FourCastNet typically takes a single state as input, or combines features
        # For simplicity, let's assume the FNO takes the last frame of the input sequence
        # and predicts the next frame as target. Or, if num_fno_blocks=1 and input_sequence_length=1
        # and output_sequence_length=1, it would be a direct frame-to-frame prediction.
        # The current FourCastNet model's forward method expects (batch, channels, H, W).
        # So, we'll extract the last frame of the input sequence and the first frame of the target sequence.

        # Adjusting input/target for a single-step prediction with the existing FourCastNet architecture
        # If you need multi-step input/output, the FNO2d and FourCastNet need modification.
        # Here, we assume a single time step input and single time step target, taken from the sequences.
        model_input = input_sequence[-1] # Take the last frame of the input sequence
        model_target = target_sequence[0] # Take the first frame of the target sequence

        return model_input, model_target