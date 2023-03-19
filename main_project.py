import tensorflow as tf
#import cv2


input_text = tf.keras.Input(shape=(None,)) 
input_image = tf.keras.Input(shape=(100, 100, 3))   #width, height, ch


#layer1: translator
layer1 = tf.keras.layers.Embedding(eng_words, output_dim=128,)(input_text)

#layer2: emotion trainer
layer2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(input_image)

'''
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

            predict_expression=model.predict(resized)

            #(audio at the same time) add later

        cv2.imshow('frame', resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release() 
cv2.destroyAllWindows() 

'''

concatenated = tf.keras.layers.Concatenate()([layer1, layer2])
output = tf.keras.layers.Dense(32, activation='softmax')(concatenated)


model = tf.keras.Model(inputs=[input_text, input_image], outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit([en_words, emotion], kor_words, epochs=100)
