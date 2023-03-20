import speech_recognition as sr
import spacy
import keras
import keras.layers
from keras.preprocessing.text import Tokenizer
from  keras_preprocessing.sequence  import  pad_sequences
import cv2
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from sklearn.model_selection import train_test_split
import os
import numpy as np


#layer1: translator

nlp = spacy.load('en_core_web_sm')

r = sr.Recognizer()

with sr.AudioFile("Audio_file.wav") as source:
    audio = r.record(source)

text = r.recognize_google(audio)

word_list=''
for words in text:
    word_list+=words


doc = nlp(word_list)

tok_list=[]
for sent in doc.sents:
    
    for token in sent:
        
        if 'CONJ' in token.pos_:
            tok_list.append("\n")

        tok_list.append(token.text)


eng_sentences=tok_list

with open('korean_sentences.txt', 'r', encoding='UTF') as f:
    kor_sentences = f.read().splitlines()



tokenizer = Tokenizer()

tokenizer.fit_on_texts(eng_sentences + kor_sentences)


eng_sequences = tokenizer.texts_to_sequences(eng_sentences)
kor_sequences  = tokenizer.texts_to_sequences(kor_sentences)

max_len = max(len(s) for s in eng_sequences + kor_sequences)

padded_eng = pad_sequences(eng_sequences, maxlen=max_len)
padded_kor  = pad_sequences(kor_sequences, maxlen=max_len)

def max_val(new_list):
    max_val = 0
    for lst in new_list:
        for x in lst:
            if x > max_val:
                max_val = x
    
    return max_val

eng_words = max_val(eng_sequences)
kor_words = max_val(kor_sequences)-eng_words


text_inputs = keras.Input(shape=(None,))
x = keras.layers.Embedding(eng_words, output_dim=128,)(text_inputs)
x = keras.layers.Dropout(rate=0.2)(x)
x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(x)
x = keras.layers.Attention()(x)
layer1 = keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.2))(x)

#layer2: emotion trainer

'''
#in the case of video
cap= cv2.VideoCapture('talking.mp4')

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():
    ret,frame= cap.read()

    if not ret:
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 3)

	     
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            
            cropped_face = gray[y:y+h, x:x+w]
            resized=cv2.resize(cropped_face, (100,100))


            #label=model.predict(resized)

           
cap.release() 
cv2.destroyAllWindows() 

'''


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



input_image = keras.Input(shape=(100, 100, 3))    #width, height, ch
x = keras.models.Sequential()(input_image)
x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1))(x)
x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
layer2 = keras.layers.Dense(10, activation='softmax')(x)


concatenated = keras.layers.Concatenate()([layer1, layer2])

outputs = keras.layers.Dense(32, activation='softmax')(concatenated)

model = keras.Model(inputs=[eng_sequences, input_image], outputs=outputs)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit([padded_eng, X_train], [padded_kor, y_train], epochs=100)
