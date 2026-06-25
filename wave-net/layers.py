import tensorflow as tf
from tensorflow.keras import layers

class DilatedCausalConv1D(layers.Layer):
  def __init__(self,filters,kernel_size,dilation_rate,**kwargs):
    super(DilatedCausalConv1D,self).__init__(**kwargs)
    self.filters=filters
    self.kernel_size=kernel_size
    self.dilation_rate=dilation_rate
    self.padding_amount=(kernel_size-1)*dilation_rate
    self.conv=layers.Conv1D(
        filters=filters,kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='valid'
    )
  def call(self,inputs):
    # Corrected padding: [[batch_before, batch_after], [seq_before, seq_after], [features_before, features_after]]
    padded_inputs=tf.pad(
        inputs,tf.constant([[0, 0], [self.padding_amount, 0], [0, 0]])
    )
    return self.conv(padded_inputs)
  def get_config(self):
    config = super(DilatedCausalConv1D, self).get_config()
    config.update({
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'dilation_rate': self.dilation_rate,
    })
    return config

class WaveNetBlock(layers.Layer):
  def __init__(self,filters,kernel_size,dilation_rate,residual_filters,skip_filters,**kwargs):
    super(WaveNetBlock,self).__init__(**kwargs)
    self.filters=filters
    self.kernel_size=kernel_size
    self.dilation_rate=dilation_rate
    self.residual_filters=residual_filters
    self.skip_filters=skip_filters
    self.conv_gate=DilatedCausalConv1D(
        filters=2*filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        name=f'dilated_conv_gate_d{dilation_rate}'
    )
    self.conv_residual=layers.Conv1D(
        filters=residual_filters,
        kernel_size=1,
        padding='valid',
        name=f'conv_residual_d{dilation_rate}' # Corrected typo here
    )
    self.conv_skip=layers.Conv1D(
        filters=skip_filters,
        kernel_size=1,
        padding='valid',
        name=f'conv_skip_d{dilation_rate}'
    )
  def call(self,inputs):
    residual_input=inputs
    gated_output=self.conv_gate(inputs)
    tanh_out=tf.tanh(gated_output[:, :, :self.filters])
    sigmoid_out=tf.sigmoid(gated_output[:, :, self.filters:])
    # Standard WaveNet gated activation is multiplication, not addition
    activated_output=tanh_out * sigmoid_out
    residual_output=self.conv_residual(activated_output)
    skip_output=self.conv_skip(activated_output)
    # Ensure residual_input and residual_output have compatible shapes for addition
    # (i.e., same number of channels). The residual_filters parameter should ensure this.
    return residual_input+residual_output,skip_output
  def get_config(self):
    config = super(WaveNetBlock, self).get_config()
    config.update({
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'dilation_rate': self.dilation_rate,
        'residual_filters': self.residual_filters,
        'skip_filters': self.skip_filters,
    })
    return config