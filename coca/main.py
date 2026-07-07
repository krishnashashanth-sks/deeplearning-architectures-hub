import tensorflow as tf
import os
import collections
import json
from model import ContrastiveCaptioner
from tokenizer import build_tokenizer_and_vocab
from dataset import create_dataset
from train import train_model
from inference import generate_caption
from layers import TextDecoder,ImageEncoder

# Let's define some constants
IMG_SIZE = 299  # InceptionV3 input size
BATCH_SIZE = 32
BUFFER_SIZE = 1000
ATTENTION_FEATURES_SHAPE = 64 # Placeholder, will be determined by CNN output

# --- Simulate Data Loading and Preprocessing ---
# In a real scenario, you would parse COCO JSON to get image paths and captions.
# We'll create a dummy dataset for now.

# Dummy image paths and captions
dummy_image_paths = [
    '/tmp/image_001.jpg',
    '/tmp/image_002.jpg',
    '/tmp/image_003.jpg',
    '/tmp/image_004.jpg',
    '/tmp/image_005.jpg'
]

dummy_captions = [
    "A cat sitting on a mat.",
    "A dog playing with a ball.",
    "Two people walking in the park.",
    "A car driving on the road at night.",
    "The sun setting over the mountains."
]

# Map each image path to all its dummy captions (one-to-one for simplicity here)
image_caption_pairs = []

# Load real COCO annotations
annotation_file_path = 'coco_dataset/annotations/annotations/captions_train2017.json'

with open(annotation_file_path, 'r') as f:
    annotations = json.load(f)

# Group captions by image ID
image_id_to_captions = collections.defaultdict(list)
for val in annotations['annotations']:
    caption = '<start> ' + val['caption'] + ' <end>'
    image_id_to_captions[val['image_id']].append(caption)

# Create a list of (image_path, caption) pairs
all_image_paths = []
all_captions = []

# Limit the number of images to process for faster demonstration
MAX_IMAGES_TO_PROCESS = 5000 # Process up to 5000 images, adjust as needed

for img_info in annotations['images']:
    image_id = img_info['id']
    if image_id in image_id_to_captions:
        image_path = os.path.join('coco_dataset/images/train2017', '%012d.jpg' % image_id)
        for caption in image_id_to_captions[image_id]:
            all_image_paths.append(image_path)
            all_captions.append(caption)
        if len(all_image_paths) >= MAX_IMAGES_TO_PROCESS: # Stop if we have enough data
            break

# Create image_caption_pairs from the real COCO data
image_caption_pairs = list(zip(all_image_paths, all_captions))

tokenizer, max_caption_length = build_tokenizer_and_vocab(all_captions)
# Instantiate the Image Encoder
EMBEDDING_DIM = 256 # This will be the dimension of our shared latent space
image_encoder = ImageEncoder(EMBEDDING_DIM)

# Decoder parameters
VOCAB_SIZE = len(tokenizer.word_index) # Update VOCAB_SIZE based on the new tokenizer
DECODER_UNITS = EMBEDDING_DIM # Number of units in the LSTM, should match EMBEDDING_DIM for contrastive loss

print(f"Current VOCAB_SIZE being used for TextDecoder: {VOCAB_SIZE}")

# Instantiate the Text Decoder
text_decoder = TextDecoder(EMBEDDING_DIM, DECODER_UNITS, VOCAB_SIZE)


# Define the optimizer
optimizer = tf.keras.optimizers.Adam()

contrastive_captioner = ContrastiveCaptioner(image_encoder, text_decoder, tokenizer, max_caption_length)

# Compile the model
contrastive_captioner.compile(optimizer=optimizer)

# Instantiate the combined model

# Now create our dummy dataset
dataset = create_dataset(image_caption_pairs, tokenizer, max_caption_length)

EPOCHS = 10 # Increased epochs for better training with real captions

# Setup for checkpointing and logging
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(image_encoder=image_encoder,
                           text_decoder=text_decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
train_model(EPOCHS,contrastive_captioner,dataset,ckpt_manager)

dummy_inference_image_paths = [
    '/tmp/inference_image_001.jpg',
    '/tmp/inference_image_002.jpg',
    '/tmp/inference_image_003.jpg',
    '/tmp/inference_image_004.jpg',
    '/tmp/inference_image_005.jpg'
]

print("Generating captions for 5 dummy photos:")
for i, img_path in enumerate(dummy_inference_image_paths):
    caption = generate_caption(img_path, contrastive_captioner, tokenizer, max_caption_length)
    print(f"\nImage {i+1}: {img_path}")
    print(f"Generated Caption: {caption}")