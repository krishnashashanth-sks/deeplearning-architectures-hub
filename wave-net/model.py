from layers import *
from tensorflow import keras
from tensorflow import layers

class WaveNet(keras.Model):
  def __init__(self,num_blocks,num_layers_per_block,filters,kernel_size,residual_filters,skip_filters,output_dim,**kwargs):
    super(WaveNet,self).__init__(**kwargs)
    self.num_blocks=num_blocks
    self.num_layers_per_block=num_layers_per_block
    self.filters=filters
    self.kernel_size=kernel_size
    self.residual_filters=residual_filters # Corrected typo and removed trailing comma
    self.skip_filters=skip_filters
    self.output_dim=output_dim
    self.initial_conv=layers.Conv1D(
        filters=residual_filters,
        kernel_size=1,
        padding='valid',
        name='initial_conv'
    )
    self.wavenet_blocks=[]
    for block_idx in range(num_blocks):
      for layer_idx in range(num_layers_per_block):
        dilation_rate=2**(layer_idx%num_layers_per_block)
        self.wavenet_blocks.append(
            WaveNetBlock(
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate, # Corrected typo here
                residual_filters=residual_filters,
                skip_filters=skip_filters,
                name=f"wavenet_block_{block_idx}_{dilation_rate}"
            )
        )
    self.final_conv1=layers.Conv1D(
        filters=skip_filters,
        kernel_size=1,
        padding='valid',
        activation='relu',
        name='final_conv1'
    )
    self.final_conv2=layers.Conv1D(
        filters=output_dim,
        kernel_size=1,
        padding='valid',
        activation='softmax', # Changed to softmax for probability distribution
        name='final_conv2' # Corrected name
    )
  def call(self,inputs):
    # Ensure input is 3D: (batch, sequence_length, 1) for raw audio
    if len(inputs.shape) == 2:
        inputs=tf.expand_dims(inputs,axis=-1)
    x=self.initial_conv(inputs)
    skip_connections=[]
    for block in self.wavenet_blocks:
      x,skip_output=block(x)
      skip_connections.append(skip_output)
    # The fix: According to the WaveNet paper, only skip connections are summed for the final layers.
    # The final 'x' (residual output) is typically discarded if not feeding another block.
    aggregated_skip=tf.add_n(skip_connections)
    x=self.final_conv1(aggregated_skip)
    x=self.final_conv2(x)
    return x
  def get_config(self):
    config = super(WaveNet, self).get_config()
    config.update({
        'num_blocks': self.num_blocks,
        'num_layers_per_block': self.num_layers_per_block,
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'residual_filters': self.residual_filters,
        'skip_filters': self.skip_filters,
        'output_dim': self.output_dim,
    })
    return config