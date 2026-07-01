class HParams:
    """Dummy Hyperparameters class for demonstration."""
    def __init__(self):
        self.n_symbols = 128  # Number of characters in the vocabulary
        self.symbols_embedding_dim = 512

        # Encoder parameters
        self.encoder_n_convolution = 3
        self.encoder_kernel_size = 5
        self.encoder_hidden_dim = 512 # Bidirectional LSTM output size

        # Attention parameters
        self.attention_rnn_dim = 1024
        self.attention_location_n_filters = 32
        self.attention_location_kernel_size = 31

        # Decoder parameters
        self.decoder_rnn_dim = 1024
        self.prenet_dims = [256, 256]
        self.n_mel_channels = 80 # Number of mel-spectrogram bins
        self.n_frames_per_step = 1 # Number of frames predicted per decoder step
        self.max_decoder_steps = 1000 # Max steps for inference
        self.gate_threshold = 0.5

        # Postnet parameters
        self.postnet_n_convolution = 5
        self.postnet_embedding_dim = 512
        self.postnet_kernel_size = 5 
        self.learning_rate = 1e-3
        self.weight_decay = 1e-6
        self.grad_clip_thresh = 1.0