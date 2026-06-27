from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Dense,Flatten
from tensorflow.keras.models import Model
import tensorflow as tf
from losses import triplet_loss

def build_siamese_net(img_height,img_width):
    input_shape=(img_height,img_width,1)
    input_tensor=Input(shape=input_shape)
    x=Conv2D(32,(3,3),activation='relu',padding='same')(input_tensor)
    x=MaxPooling2D((2,2))(x)
    x=Conv2D(64,(3,3),activation='relu',padding='same')(x)
    x=MaxPooling2D((2,2))(x)
    x=Conv2D(128,(3,3),activation='relu',padding='same')(x)
    x=Flatten()(x)
    x=Dense(128,activation='relu')(x)
    embedding=Dense(128)(x)
    embedding_model=Model(input_tensor,embedding,name="embedding_model")
    # Create the input tensors for the anchor, positive, and negative images
    anchor_input = Input(shape=input_shape, name="anchor")
    positive_input = Input(shape=input_shape, name="positive")
    negative_input = Input(shape=input_shape, name="negative")

    # Get the embeddings for each input using the shared embedding_model
    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    # Create the Siamese network model
    return  Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=[anchor_embedding, positive_embedding, negative_embedding]
    )

class SiameseModel(tf.keras.Model):
    def __init__(self, siamese_network, margin=0.2):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # data is a batch of triplets: (anchor, positive, negative)
        with tf.GradientTape() as tape:
            anchor, positive, negative = data[:, 0], data[:, 1], data[:, 2]

            # Forward pass through the siamese network
            anchor_embedding, positive_embedding, negative_embedding = self.siamese_network(
                [anchor, positive, negative]
            )

            # Calculate triplet loss
            loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding, self.margin)
            loss = tf.reduce_mean(loss)

        # Compute gradients and apply them
        gradients = tape.gradient(loss, self.siamese_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_variables))

        # Update loss tracker
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]
