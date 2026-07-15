from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense,BatchNormalization,Dropout
from tensorflow.keras.models import Model
def build_model(img_shape=(64,64,3),num_classes=10):
    input_layer=Input(shape=img_shape,name="input_image")
    x=Conv2D(32,(3,3),activation='relu',padding='same',name="conv1")(input_layer)
    x=BatchNormalization(name='batch_norm1')(x)
    x=MaxPooling2D((2,2),name="pool1")(x)
    x=Dropout(0.25,name='dropout1')(x)

    x=Conv2D(64,(3,3),activation='relu',padding='same',name="conv2")(x)
    x=BatchNormalization(name='batch_norm2')(x)
    x=MaxPooling2D((2,2),name="pool2")(x)
    x=Dropout(0.25,name='dropout2')(x)

    x=Conv2D(128,(3,3),activation='relu',padding='same',name="conv3")(x)
    x=BatchNormalization(name='batch_norm3')(x)
    x=MaxPooling2D((2,2),name="pool3")(x)
    x=Dropout(0.25,name='dropout3')(x)

    x=Flatten(name="flatten")(x)
    x=Dense(256,activation='relu',name='dense1')(x)
    x=BatchNormalization(name="batch_norm_dens1")(x)
    x=Dropout(0.5,name="dropout-dense1")(x)
    output_layer=Dense(num_classes,activation='softmax',name="output_classes")(x)
    return Model(inputs=input_layer,outputs=output_layer,name="onn_model")