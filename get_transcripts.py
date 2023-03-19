import speech_recognition as sr
import spacy
import re
from keras.models import load_model
from keras.layers import *

nlp = spacy.load('en_core_web_sm')

#get transcripts from audios
r = sr.Recognizer()

with sr.AudioFile("Audio_file.wav") as source:
    audio = r.record(source)

text = r.recognize_google(audio)

word_list=''
for words in text:
    word_list+=words


#sentence division

doc = nlp(word_list)

tok_list=[]
for sent in doc.sents:
    
    for token in sent:
        
        if 'CONJ' in token.pos_:
            tok_list.append("\n")

        tok_list.append(token.text)
    
tok_sen=' '.join([tok for tok in tok_list])

tok_sen=re.sub(r' \n ', '\n', tok_sen)

f1=open('transcript_1.txt', 'w', encoding="utf-8")
f1.write(tok_sen)


#translate

model=load_model('model_name.hdf5')
result=model.predict(tok_sen)


