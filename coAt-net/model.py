from layers import *

class CoAtNet(tf.keras.Model):
  def __init__(self,num_classes,input_shape=(224,224,3),num_blocks_per_stage=[2,2,3,5,2],
               filters_per_stage=[64,96,192,384,768],transformer_stages=[2,3,4], # `transformer_stages` refers to 1-indexed stage numbers
               embed_dim_per_stage=[192,384,768],num_heads_per_stage=[4,8,16],
               ff_dim_per_stage=[768,1536,3071],**kwargs):
    super().__init__(**kwargs)
    self.input_shape = input_shape # Store input_shape as an attribute
    self.stem=tf.keras.Sequential([
        layers.Conv2D(filters_per_stage[0]//2,kernel_size=3,strides=2,padding='same',use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('swish'),
        layers.Conv2D(filters_per_stage[0],kernel_size=3,strides=2,padding='same',use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('swish')] # Removed input_shape as it's not needed here
    )
    self.stages_config = [] # Store stage configurations
    current_channels = filters_per_stage[0] # Output channels of the stem

    transformer_config_idx = 0 # Index for embed_dim_per_stage, num_heads_per_stage, ff_dim_per_stage

    for i in range(len(num_blocks_per_stage)):
      stage_num = i + 1 # 1-indexed stage number
      is_transformer = stage_num in transformer_stages

      stage_info = {'type': '', 'layer': None, 'projection': None, 'reshape_to_3d': False, 'reshape_to_4d': False}

      if is_transformer:
        stage_info['type'] = 'transformer'

        # Get transformer specific parameters
        actual_embed_dim = embed_dim_per_stage[transformer_config_idx]
        actual_num_heads = num_heads_per_stage[transformer_config_idx]
        actual_ff_dim = ff_dim_per_stage[transformer_config_idx]
        transformer_config_idx += 1

        # Add a projection if the number of channels from the previous stage
        # does not match the embed_dim required for the transformer.
        if current_channels != actual_embed_dim:
            if i > 0 and self.stages_config[i-1]['type'] == 'transformer': # Previous was transformer, so current input x is 3D
                stage_info['projection'] = layers.Dense(actual_embed_dim, name=f'stage_{stage_num}_transformer_dense_proj')
            else: # Previous was conv or stem, so current input x is 4D
                stage_info['projection'] = layers.Conv2D(actual_embed_dim, kernel_size=1, padding='same', use_bias=False, name=f'stage_{stage_num}_transformer_conv_proj')

        stage_info['layer'] = CoAtNetStage(
            num_blocks=num_blocks_per_stage[i],
            filters_in=current_channels, # This is for type hint in CoAtNetStage, not used by TransformerBlock
            filters_out=actual_embed_dim, # This is for type hint in CoAtNetStage, not used by TransformerBlock
            block_type='transformer',
            embed_dim=actual_embed_dim,
            num_heads=actual_num_heads,
            ff_dim=actual_ff_dim,
            name=f'CoAtNetStage_{stage_num}_Transformer'
        )
        stage_info['reshape_to_3d'] = True # Always reshape to 3D before transformer stage
        current_channels = actual_embed_dim # Output channels of transformer stage
      else: # Convolutional stage
        stage_info['type'] = 'conv'

        # Determine stride for MBConv block: first conv stage after stem (i=0) has stride 1, others stride 2.
        # This aligns with common CoAtNet structure where stem does initial downsampling.
        stage_stride = 1 if i == 0 else 2

        # If previous stage was transformer, mark that we need to reshape from 3D to 4D.
        if i > 0 and self.stages_config[i-1]['type'] == 'transformer':
            stage_info['reshape_to_4d'] = True

        stage_info['layer'] = CoAtNetStage(
            num_blocks=num_blocks_per_stage[i],
            filters_in=current_channels,
            filters_out=filters_per_stage[i],
            block_type='conv',
            stride=stage_stride,
            kernel_size=3,
            expand_ratio=6,
            se_ratio=0.25,
            activation='swish',
            name=f'CoAtNetStage_{stage_num}_Conv'
        )
        current_channels = filters_per_stage[i] # Output channels of conv stage

      self.stages_config.append(stage_info)

    # Final layers, outside the loop
    self.avg_pool=layers.GlobalAveragePooling2D(name='global_average_pooling')
    self.classifier=layers.Dense(num_classes,activation='softmax', name='classifier_head')

  def call(self,inputs,training=False):
      x=self.stem(inputs) # Output is 4D (batch, H, W, C)

      for stage_info in self.stages_config:
        stage_type = stage_info['type']
        stage_layer = stage_info['layer']
        projection_layer = stage_info['projection']
        reshape_to_3d = stage_info['reshape_to_3d']
        reshape_to_4d = stage_info['reshape_to_4d']

        if reshape_to_4d: # If current is conv and previous was transformer, reshape from 3D to 4D
            # x is (batch, seq_len, embed_dim)
            current_shape = tf.shape(x) # Use tf.shape for dynamic shapes
            batch_size = current_shape[0]
            seq_len = current_shape[1]
            embed_dim = current_shape[2]

            spatial_dim = tf.cast(tf.math.sqrt(tf.cast(seq_len, tf.float32)), tf.int32)
            # Add a check for perfect square only if static shape is available
            # if tf.debugging.is_nan(spatial_dim) or spatial_dim * spatial_dim != seq_len:
            #     raise ValueError(f"Cannot reshape from 3D to 4D: Sequence length {seq_len} is not a perfect square.")
            x = tf.reshape(x, (batch_size, spatial_dim, spatial_dim, embed_dim))

        if projection_layer: # Apply projection if defined. It could be Conv2D (if prev was conv) or Dense (if prev was transformer)
            x = projection_layer(x)

        if reshape_to_3d: # If current is transformer and input is 4D, flatten to 3D
            if len(x.shape) == 4: # Only flatten if currently 4D
                current_shape = tf.shape(x)
                batch_size,height,width,channels=current_shape[0], current_shape[1], current_shape[2], current_shape[3]
                x=tf.reshape(x,(-1,height*width,channels))
            # else, if already 3D, no need to reshape (e.g. transformer following transformer)

        x = stage_layer(x, training=training) # Apply the stage layer and pass training

      # Final pooling and classification
      if len(x.shape)==4: # If the last stage's output is 4D (e.g., last stage was conv)
        x=self.avg_pool(x)
      elif len(x.shape)==3: # If the last stage's output is 3D (e.g., last stage was transformer)
        x=tf.reduce_mean(x,axis=1) # Global average pooling over the sequence dimension (H*W)

      return self.classifier(x)