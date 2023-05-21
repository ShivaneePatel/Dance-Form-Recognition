import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

dataset_path = 'dataset/train'
class_names = os.listdir(dataset_path)
# class_names.append('other')  # Add an extra class label for videos that do not match any of the other classes

def extract_frames(video_path, output_path,start_frame=10, end_frame=15):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(round(fps))
    frame_count = 1  
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % interval == 0:
                output_file = os.path.join(output_path, f"frame_{frame_count}.jpg")
                cv2.imwrite(output_file, frame)
            frame_count += 1

    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")       
    finally:
        cap.release()
for class_name in class_names:
    class_path = os.path.join(dataset_path, class_name)
    video_files = os.listdir(class_path)
    for video_file in video_files:
        video_path = os.path.join(class_path, video_file)
        output_path = os.path.join(class_path, video_file[:-4])
        os.makedirs(output_path, exist_ok=True)
        extract_frames(video_path, output_path)

img_size = (128, 128)
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(dataset_path, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='training')
validation_generator = train_datagen.flow_from_directory(dataset_path, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='validation')

model = Sequential()

# CNN layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())

# LSTM layer
model.add(Reshape((1, -1)))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(128))

# Output layer
model.add(Dense(len(class_names), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, steps_per_epoch=train_generator.samples//train_generator.batch_size, validation_data=validation_generator, validation_steps=validation_generator.samples//validation_generator.batch_size, epochs=10)

# Extract accuracy and loss data from history object
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot accuracy data
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Add text annotations for percentage of accuracy
for i in range(len(acc)):
    plt.annotate('{:.2%}'.format(acc[i]), xy=(epochs[i], acc[i]), ha='center', va='bottom')
    plt.annotate('{:.2%}'.format(val_acc[i]), xy=(epochs[i], val_acc[i]), ha='center', va='top')
    
plt.show()

model.save('dance_classification_model.h5')