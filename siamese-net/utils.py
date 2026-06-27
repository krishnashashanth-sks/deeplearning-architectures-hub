import numpy as np

def create_triplets(images, labels, num_triplets=10000):
    triplets = []
    # Group images by label for efficient sampling
    label_to_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}

    for _ in range(num_triplets):
        # Select an anchor class and an anchor image
        anchor_label = np.random.choice(np.unique(labels))
        anchor_index = np.random.choice(label_to_indices[anchor_label])
        anchor_image = images[anchor_index]

        # Select a positive image from the same class
        positive_index = np.random.choice(label_to_indices[anchor_label])
        while positive_index == anchor_index:
            positive_index = np.random.choice(label_to_indices[anchor_label])
        positive_image = images[positive_index]

        # Select a negative class (different from anchor class) and a negative image
        negative_label = np.random.choice([l for l in np.unique(labels) if l != anchor_label])
        negative_index = np.random.choice(label_to_indices[negative_label])
        negative_image = images[negative_index]

        triplets.append((anchor_image, positive_image, negative_image))

    return np.array(triplets)