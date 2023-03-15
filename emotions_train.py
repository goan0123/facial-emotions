import cv2
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
import numpy as np

input_dir = 'data'
emotions=['Angry', 'Depressed', 'Disgust', 'Excited', 'Fear', 'Frustrated', 'Happy','Neutral','Sad','Surprise']

data = []
labels = []


for i, emotion in enumerate(emotions):
    for file in os.listdir(os.path.join(input_dir, emotion)):
        img_path = os.path.join(input_dir, emotion, file)

        img = cv2.imread(img_path)
        img = cv2.resize(img, (48, 48))

        data.append(img.flatten())
        labels.append(i)


data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))



model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])


#model.save('model_name.hdf5')

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
modelpath="/model_name.hdf5"        
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)


history=model.fit(X_train, y_train, validation_split=0.25, epochs=100, batch_size=32, verbose=0, callbacks=[early_stopping_callback,checkpointer])
