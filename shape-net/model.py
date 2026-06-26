import tensorflow as tf

class TFShapeNetGAN(tf.keras.Model):
    def __init__(self, encoder, decoder, discriminator, **kwargs):
        super(TFShapeNetGAN, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def compile(self, generator_optimizer, discriminator_optimizer,
                 reconstruction_loss_fn, generator_loss_fn, discriminator_loss_fn,
                 reconstruction_loss_weight=1.0, generator_adversarial_weight=0.1):
        super(TFShapeNetGAN, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.generator_loss_fn = generator_loss_fn
        self.discriminator_loss_fn = discriminator_loss_fn
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.generator_adversarial_weight = generator_adversarial_weight

    @tf.function
    def train_step(self, real_point_clouds):
        if isinstance(real_point_clouds, tuple):
            real_point_clouds = real_point_clouds[0]

        # --- 1. Train Discriminator ---
        with tf.GradientTape() as tape_d:
            # Generate fake point clouds (Encoder + Decoder act as Generator)
            latent_vector = self.encoder(real_point_clouds, training=True)
            fake_point_clouds = self.decoder(latent_vector, training=True)

            # Get discriminator predictions for real and fake point clouds
            real_predictions = self.discriminator(real_point_clouds[:, :, :3], training=True) # Discriminator expects 3D coords
            fake_predictions = self.discriminator(fake_point_clouds, training=True)

            # Calculate discriminator loss
            d_loss = self.discriminator_loss_fn(real_predictions, fake_predictions)

        # Apply gradients to discriminator
        d_grads = tape_d.gradient(d_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # --- 2. Train Generator (Encoder + Decoder) ---
        with tf.GradientTape() as tape_g:
            latent_vector = self.encoder(real_point_clouds, training=True)
            reconstructed_pc = self.decoder(latent_vector, training=True)

            # Reconstruction Loss (comparing XYZ of real with reconstructed)
            reconstruction_loss = self.reconstruction_loss_fn(real_point_clouds[:, :, :3], reconstructed_pc)

            # Get discriminator predictions for generated (fake) point clouds
            # Discriminator is not trained here, so training=False
            fake_predictions_for_g = self.discriminator(reconstructed_pc, training=False)

            # Generator's adversarial loss (generator wants D to classify fake as real)
            g_adv_loss = self.generator_loss_fn(fake_predictions_for_g)

            # Total Generator Loss
            g_loss = (self.reconstruction_loss_weight * reconstruction_loss) + \
                     (self.generator_adversarial_weight * g_adv_loss)

        # Apply gradients to generator (encoder and decoder)
        g_grads = tape_g.gradient(g_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(g_grads, self.encoder.trainable_variables + self.decoder.trainable_variables))

        return {
            'd_loss': d_loss,
            'g_loss': g_loss,
            'reconstruction_loss': reconstruction_loss,
            'g_adv_loss': g_adv_loss
        }