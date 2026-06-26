import tensorflow as tf

def train(epochs, dataloader, shape_net_gan):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        d_losses = []
        g_losses = []
        recon_losses = []
        g_adv_losses = []

        # Iterate through the dataloader to get real point clouds
        for i, real_point_clouds in enumerate(dataloader):
            # Perform one training step
            metrics = shape_net_gan.train_step(real_point_clouds)

            d_losses.append(metrics['d_loss'].numpy())
            g_losses.append(metrics['g_loss'].numpy())
            recon_losses.append(metrics['reconstruction_loss'].numpy())
            g_adv_losses.append(metrics['g_adv_loss'].numpy())

            # Assuming dataloader has a known length or providing frequent updates
            if i % 10 == 0: # Print progress every few batches
                print(f"Batch {i} - "
                      f"D Loss: {metrics['d_loss'].numpy():.4f}, "
                      f"G Loss: {metrics['g_loss'].numpy():.4f}, "
                      f"Recon Loss: {metrics['reconstruction_loss'].numpy():.4f}")

        # Calculate average metrics for the epoch
        avg_d_loss = tf.reduce_mean(d_losses)
        avg_g_loss = tf.reduce_mean(g_losses)
        avg_recon_loss = tf.reduce_mean(recon_losses)
        avg_g_adv_loss = tf.reduce_mean(g_adv_losses)

        print(f"Epoch {epoch + 1} Summary: "
              f"Avg D Loss: {avg_d_loss:.4f}, "
              f"Avg G Loss: {avg_g_loss:.4f}, "
              f"Avg Recon Loss: {avg_recon_loss:.4f}, "
              f"Avg G Adv Loss: {avg_g_adv_loss:.4f}")
    return d_losses, g_losses