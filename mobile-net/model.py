import tensorflow as tf
from layers import _inverted_res_block
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, ZeroPadding2D, GlobalAveragePooling2D, Dense

def MobileNetV2(input_shape=(224,224,3),num_classes=1000,alpha=1.0):
    img_input=tf.keras.Input(shape=input_shape)
    x=ZeroPadding2D(padding=((0,1),(0,1)),name='conv1_pad')(img_input)
    x=Conv2D(int(32*alpha),kernel_size=3,strides=(2,2),padding='valid',use_bias=False,name='conv1')(x) # Fixed kernel_size and called the layer
    x=BatchNormalization(name='bn_conv1')(x)
    x=ReLU(6.,name='conv1_relu')(x)
    x=_inverted_res_block(x,expansion=1,stride=1,filters=16,block_id=0,alpha=alpha)

    x=_inverted_res_block(x,expansion=6,stride=2,filters=24,block_id=1,alpha=alpha) # Corrected expansion
    x=_inverted_res_block(x,expansion=6,stride=1,filters=24,block_id=2,alpha=alpha) # Corrected expansion

    x=_inverted_res_block(x,expansion=6,stride=2,filters=32,block_id=3,alpha=alpha) # Corrected expansion
    x=_inverted_res_block(x,expansion=6,stride=1,filters=32,block_id=4,alpha=alpha) # Corrected expansion
    x=_inverted_res_block(x,expansion=6,stride=1,filters=32,block_id=5,alpha=alpha) # Corrected expansion

    x=_inverted_res_block(x,expansion=6,stride=2,filters=64,block_id=6,alpha=alpha) # Corrected expansion
    x=_inverted_res_block(x,expansion=6,stride=1,filters=64,block_id=7,alpha=alpha) # Corrected expansion
    x=_inverted_res_block(x,expansion=6,stride=1,filters=64,block_id=8,alpha=alpha) # Corrected expansion
    x=_inverted_res_block(x,expansion=6,stride=1,filters=64,block_id=9,alpha=alpha) # Corrected expansion

    x=_inverted_res_block(x,expansion=6,stride=1,filters=96,block_id=10,alpha=alpha) # Corrected expansion
    x=_inverted_res_block(x,expansion=6,stride=1,filters=96,block_id=11,alpha=alpha) # Corrected expansion
    x=_inverted_res_block(x,expansion=6,stride=1,filters=96,block_id=12,alpha=alpha) # Corrected expansion

    x=_inverted_res_block(x,expansion=6,stride=2,filters=160,block_id=13,alpha=alpha) # Corrected expansion
    x=_inverted_res_block(x,expansion=6,stride=1,filters=160,block_id=14,alpha=alpha) # Corrected expansion
    x=_inverted_res_block(x,expansion=6,stride=1,filters=160,block_id=15,alpha=alpha) # Corrected expansion

    x=_inverted_res_block(x,expansion=6,stride=1,filters=320,block_id=16,alpha=alpha) # Corrected expansion
    x=Conv2D(int(1280*alpha) if alpha > 1.0 else 1280,kernel_size=1,use_bias=False,name='conv_1')(x) # Fixed final channel count for alpha <= 1.0
    x=BatchNormalization(name='conv_1_bn')(x)
    x=ReLU(6.,name='conv_1_relu')(x)
    x=GlobalAveragePooling2D(name='global_average_pooling')(x)
    x=Dense(num_classes,activation='softmax',name='predictions')(x)
    model=Model(img_input,x,name=f'mobilenet_v2_{alpha:.1f}')
    return model
