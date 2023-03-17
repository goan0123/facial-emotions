import keras
import keras.layers
from keras.preprocessing.text import Tokenizer
from  keras_preprocessing.sequence  import  pad_sequences


with open('english_sentences.txt', 'r', encoding='UTF') as f:
    eng_sentences = f.read().splitlines()

with open('korean_sentences.txt', 'r', encoding='UTF') as f:
    kor_sentences = f.read().splitlines()



tokenizer = Tokenizer()

tokenizer.fit_on_texts(eng_sentences + kor_sentences)

# preprocessing: Generate sequences from the tokens and pad them
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


inputs = keras.Input(shape=(None,))
x = keras.layers.Embedding(eng_words, output_dim=128,)(inputs)
x = keras.layers.Dropout(rate=0.2)(x)
x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(x)
x = keras.layers.Attention()(x)
x = keras.layers.Dropout(rate=0.2)(x)
outputs = keras.layers.TimeDistributed(keras.layers.Dense(kor_words, activation='softmax'))(x)


model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(padded_eng, padded_kor, epochs=100, batch_size=32)
