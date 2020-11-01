import tensorflow.keras as keras
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

optimizer = optimizers.Adam(lr=0.0005)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=len(train_data_gen) / batch_size,
    epochs=100,
    validation_data=val_data_gen,
    validation_steps=len(val_data_gen) / batch_size
)

model.summary()

_, model_accuracy = model.evaluate_generator(val_data_gen)
print('TF Model accuracy:', model_accuracy)

model.save('./model/Intel_base_model.h5')