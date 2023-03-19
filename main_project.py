import tensorflow as tf



input_text = tf.keras.Input(shape=(None,)) 
input_image = tf.keras.Input(shape=(100, 100, 3))   #width, height, ch


#layer1: translator
layer1 = tf.keras.layers.Embedding(eng_words, output_dim=128,)(input_text)

#layer2: emotion trainer
layer2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(input_image)


concatenated = tf.keras.layers.Concatenate()([layer1, layer2])
output = tf.keras.layers.Dense(32, activation='softmax')(concatenated)


model = tf.keras.Model(inputs=[input_text, input_image], outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit([en_words, emotion], kor_words, epochs=100)