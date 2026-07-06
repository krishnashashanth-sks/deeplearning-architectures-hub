from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model

def build_model(input_shape_stream1=(64,64,3),input_shape_stream2=(32,32,1)):
    
    input_stream1=Input(shape=input_shape_stream1,name='input_stream1')
    input_stream2=Input(shape=input_shape_stream2,name='input_stream2')

    x=Conv2D(32,(3,3),activation='relu',padding='same')(input_stream1)
    x=MaxPooling2D((2,2))(x)
    x=Conv2D(64,(3,3),activation='relu',padding='same')(x)
    x=MaxPooling2D((2,2))(x)
    x=Flatten()(x)
    stream1_output=Dense(128,activation='relu')(x)

    y=Conv2D(26,(3,3),activation='relu',padding='same')(input_stream2)
    y=MaxPooling2D((2,2))(y)
    x=Conv2D(32,(3,3),activation='relu',padding='same')(y)
    y=MaxPooling2D((2,2))(y)
    y=Flatten()(y)
    stream2_output=Dense(64,activation='relu')(y)
    merged=concatenate([stream1_output,stream2_output])
    z=Dense(256,activation='relu')(merged)
    output_layer=Dense(10,activation='softmax')(z)
    return Model(inputs=[input_stream1,input_stream2],outputs=output_layer)