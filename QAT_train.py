from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

# setting the train, test and val directories
train_dir = r'.\archive\seg_train\seg_train'
test_dir = r'.\archive\seg_pred'
val_dir = r'.\archive\seg_test\seg_test'

# setting basic parameters to the model

IMG_WIDTH = 100
IMG_HEIGHT = 100
IMG_DIM = (IMG_HEIGHT, IMG_WIDTH)
batch_size = 16
epochs = 25

# creating Image Data generator
image_gen_train = ImageDataGenerator(rescale=1. / 255,
                                     zoom_range=0.3,
                                     rotation_range=25,
                                     shear_range=0.1,
                                     featurewise_std_normalization=False)

# Creating train data generator
train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=IMG_DIM,
                                                     class_mode='sparse')

# Creating validation data generator
image_gen_val = ImageDataGenerator(rescale=1. / 255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=val_dir,
                                                 target_size=IMG_DIM,
                                                 class_mode='sparse')

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    keras.layers.Reshape(target_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.3),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.3),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(6, activation='softmax')
])

import tensorflow_model_optimization as tfmot
import tensorflow as tf

quantize_model = tfmot.quantization.keras.quantize_model
# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)
optimizer = optimizers.Adam(lr=0.0004)
# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

history = q_aware_model.fit_generator(
    train_data_gen,
    steps_per_epoch=len(train_data_gen) / batch_size,
    epochs=100,
    validation_data=val_data_gen,
    validation_steps=len(val_data_gen) / batch_size
)

# print network structure
q_aware_model.summary()

# evaluate trained model
_, qat_model_accuracy = q_aware_model.evaluate_generator(val_data_gen)
print('Quant Model accuracy:', qat_model_accuracy)

# save trained model
q_aware_model.save('./model/Intel_quantize_aware_model.h5')

# convert TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()
with open('./model/Intel_QAT.tflite', 'wb') as f:
    f.write(quantized_tflite_model)

# inference test
import numpy as np

class_names = list(train_data_gen.class_indices.keys())
# Create the interpreter for the TfLite model
interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
interpreter.allocate_tensors()
# Create input and output tensors from the interpreter
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
# Create the image data for prediction
dataset_list = tf.data.Dataset.list_files(test_dir + '\\*')
for i in range(10):
    image = next(iter(dataset_list))
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (100, 100))
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)

    # Set the tensor for image into input index
    interpreter.set_tensor(input_index, image)

    # Run inference.
    interpreter.invoke()
    # find the prediction with highest probability.
    output = interpreter.tensor(output_index)
    pred = np.argmax(output()[0])

    print(class_names[pred])