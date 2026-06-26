from inference import infer_advanced_model
from train import train
from losses import *
from encoder import TFAdvanced3DEncoder
from decoder import TFFoldingDecoder
from discriminator import TFDiscriminator
from model import TFShapeNetGAN
import tensorflow as tf
from visualize import visualize_point_cloud

input_feature_dim_example = 6 # (x,y,z) + (nx,ny,nz)
latent_dim_example = 256
num_points_example = 1024
output_points_example = 1024
grid_res_example = 32

tf_encoder_gan = TFAdvanced3DEncoder(
    input_feature_dim=input_feature_dim_example,
    latent_dim=latent_dim_example,
    num_points=num_points_example
)
tf_decoder_gan = TFFoldingDecoder(
    latent_dim=latent_dim_example,
    output_num_points=output_points_example,
    grid_res=grid_res_example
)
tf_discriminator_gan = TFDiscriminator(
    input_shape=(num_points_example, 3) # Discriminator typically takes only XYZ coordinates
)

# Instantiate the full TFShapeNetGAN model
shape_net_gan = TFShapeNetGAN(
    encoder=tf_encoder_gan,
    decoder=tf_decoder_gan,
    discriminator=tf_discriminator_gan
)

# Define Optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)


shape_net_gan.compile(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    reconstruction_loss_fn=chamfer_distance, # Using the new Chamfer Distance
    generator_loss_fn=dummy_generator_loss,
    discriminator_loss_fn=dummy_discriminator_loss,
    reconstruction_loss_weight=100.0, # Often Chamfer is weighted higher
    generator_adversarial_weight=1.0 
)

batch_size=1

# Define parameters for the dummy dataset
num_samples_in_dataset = 1000

# Function to generate a single dummy point cloud
def generate_dummy_point_cloud():
    return tf.random.normal((num_points_example, input_feature_dim_example), dtype=tf.float32)

# Create a tf.data.Dataset from a generator
dummy_dataloader = tf.data.Dataset.from_generator(
    generate_dummy_point_cloud,
    output_signature=tf.TensorSpec(shape=(num_points_example, input_feature_dim_example), dtype=tf.float32)
)

# Repeat, shuffle, and batch the dataset
dummy_dataloader = dummy_dataloader.repeat().shuffle(buffer_size=100).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Verify the output shape of one batch
for example_batch in dummy_dataloader.take(1):
    print(f"\nShape of one batch from dummy_dataloader: {example_batch.shape}")

print("Dummy dataloader created successfully!")

# --- Training Hyperparameters ---
epochs = 5
# In a real scenario, this would be tf.data.Dataset or similar.
# For demonstration, we'll simulate batches.
num_dummy_batches_per_epoch = 10
batch_size = 2 # Matches the dummy_input_encoder batch size

d_losses,g_losses=train(epochs,dummy_dataloader,shape_net_gan)

reconstructed_pcs=infer_advanced_model(tf_encoder_gan,tf_decoder_gan, 5,num_points_example,input_feature_dim_example)

visualize_point_cloud(reconstructed_pcs[0], title="First Reconstructed TensorFlow Point Cloud")