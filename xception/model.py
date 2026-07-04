from tensorflow.keras import layers,Model
from layers import xception_entry_flow,xception_middle_flow,xception_exit_flow

def build_xception(input_shape=(229,299,3),num_classes=1000):
  inputs=layers.Input(shape=input_shape)
  x=xception_entry_flow(inputs)
  x=xception_middle_flow(x)
  outputs=xception_exit_flow(x,num_classes)
  return Model(inputs,outputs,name='xception_from_scratch')