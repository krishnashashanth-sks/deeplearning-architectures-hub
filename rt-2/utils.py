import numpy as np

# 2. Verify Data Generation and Dataset Creation (already confirmed in previous steps)
def generate_dummy_data(num_samples=100, img_height=128, img_width=128, img_channels=3,
                        sequence_length=50, vocab_size=10000, action_dim=7):
    dummy_images = np.random.rand(num_samples, img_height, img_width, img_channels).astype(np.float32)
    dummy_languages = np.random.randint(0, vocab_size, (num_samples, sequence_length), dtype=np.int32)
    dummy_actions = np.random.rand(num_samples, action_dim).astype(np.float32)
    return (dummy_images, dummy_languages), dummy_actions
