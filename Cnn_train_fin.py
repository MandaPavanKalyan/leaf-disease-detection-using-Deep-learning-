# -*- coding: utf-8 -*-
import tensorflow 
from keras.models import Sequential
#initialize nn
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
#convert pooling features space to large feature vector for fully
#connected layer 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense

from keras.layers import BatchNormalization
from keras.layers import Dropout
import seaborn as sns
from matplotlib import pyplot as plt



#basic cnn
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(128,128, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(25, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



train_datagen = ImageDataGenerator(rescale = None,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'D:\leaffinal\dataset\train',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
#print(test_datagen);
labels = (training_set.class_indices)
print(labels)

test_set = test_datagen.flow_from_directory(r'D:\leaffinal\dataset\test',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'categorical')

labels2 = (test_set.class_indices)
print(labels2)

history=model.fit_generator(training_set,
                         steps_per_epoch = 375,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 125)


# Part 3 - Making new predictions

model_json=model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
    model.save_weights("leaf_model1.h5")
    print("Saved model to disk")


'''train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['acc']
val_acc = history.history['val_acc']

sns.set_theme()

sns.lineplot(data=train_loss, color='b', label='train')
sns.lineplot(data=val_loss, color='r', label='val')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

sns.lineplot(data=train_acc, color='b', label='train')
sns.lineplot(data=val_acc, color='r', label='val')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()'''




