
"""# Miscellaneous"""

img_height = 224
img_width = 224
img_channels = 3

test_fraction = 0.2

"""# Preprocessing of captions"""

import pandas as pd
import numpy as np

df = pd.read_csv('train/captions.txt')
df = df[df.caption.notnull()]
captions = df[['caption']].to_numpy().flatten()
stop_word = 'STOPWORD'
captions = [x + ' ' + stop_word for x in captions]
#print(captions)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

t = Tokenizer()
t.fit_on_texts(captions)
num_words = len(t.word_index) + 1
seq = t.texts_to_sequences(captions)

# Maximum length of any sequence, plus one stop word
maxlen = max([len(x) for x in seq])

padded_captions = pad_sequences(seq, maxlen=maxlen, padding='post')
#print(padded_captions)

stop_word_idx = t.texts_to_sequences([stop_word])[0][0]

maxlen

"""# Preprocessing of images"""

filenames = df[['image']].to_numpy().flatten()

import tensorflow as tf
from tensorflow.keras.preprocessing import image

label_captions = padded_captions
input_captions = padded_captions
for caption in input_captions:
  caption[caption==stop_word_idx] = 0
input_captions = input_captions[:,:-1]
dataset = tf.data.Dataset.from_tensor_slices((filenames, input_captions, label_captions))
dataset = dataset.map(lambda x, y, z: (tf.io.read_file('train/Images/' + x), y, z))
dataset = dataset.map(lambda x, y, z: (tf.image.decode_jpeg(x, channels=img_channels), y, z))
dataset = dataset.map(lambda x, y, z: (tf.image.resize(x,[img_height, img_width]), y, z))
dataset = dataset.map(lambda x, y, z: ({'input_1':x, 'input_2':y}, z))
#dataset = dataset.batch(32, drop_remainder=True)
n_data = padded_captions.shape[0]

test_set = dataset.take(int(test_fraction*n_data)).shuffle(1000).batch(32)
train_set = dataset.skip(int(test_fraction*n_data)).shuffle(1000).batch(32)

"""# Model"""



"""# Load pre-trained VGG16 model"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model

# VGG16
vgg16_conv = vgg16.VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(img_height, img_width, img_channels))

img_input = keras.layers.Input(shape=(img_height,img_width,img_channels))
vgg16_conv.trainable = False

# We only change the activation of the last layer from softmax to relu
vgg16_conv.layers[-1].activation = keras.activations.relu
vgg16_conv.compile(loss="mse") 
#vgg16_conv.summary()
#print(vgg16_conv.get_config())
output_vgg16_conv = vgg16_conv(img_input)

# Turn output of VGG16 into sequence of 1 time step

# If 'include_top' is False
cnn_global_pooling = keras.layers.GlobalMaxPooling2D()(output_vgg16_conv)
cnn_seq = keras.layers.Reshape(target_shape=(1,512))(cnn_global_pooling)

#cnn_dense = keras.layers.Dense(512)(output_vgg16_conv)
#cnn_seq = keras.layers.Reshape(target_shape=(1,512))(cnn_dense)

class ConstantMask(keras.layers.Layer):
    def call(self, inputs):
        return inputs
    
    def compute_mask(self, inputs, mask=None):
        return tf.fill(tf.shape(inputs[:,:,0]), True)

#cnn_seq_mask = keras.layers.Masking(mask_value=np.nan)(cnn_seq)
cnn_seq_mask = ConstantMask()(cnn_seq)


# RNN input from training data (from tokenizer)
caption_input = keras.layers.Input(shape=(maxlen-1))
rnn_embed = keras.layers.Embedding(input_dim=num_words, output_dim=512, mask_zero=True)(caption_input)

# Concatenate CNN output and embedding output
merged = keras.layers.Concatenate(axis=1)([cnn_seq_mask, rnn_embed])

# LSTM layer with merged sequence as input
# For dropout, see [34], it is unclear if the parameter 'dropout' was used in the original paper
# Not sure how "ensembling" is used...
lstm = keras.layers.LSTM(512, recurrent_dropout = 0.5, return_sequences=True)(merged)
lstm_output = keras.layers.Dense(num_words, activation='softmax')(lstm)
#cropping = tf.keras.layers.Cropping1D(cropping=(0,1))(lstm_output)
model = Model(inputs=[img_input, caption_input], outputs=lstm_output)
opt = tf.keras.optimizers.SGD(learning_rate=0.01, 
                              momentum=0.0, 
                              nesterov=False, 
                              name="SGD") # See section 4.3.1

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt, 
              metrics=['accuracy'])

model.summary()

model.load_weights('./captioning_model/2020-11-17-10:04/model.h5')
#model.fit(train_set, epochs=20)
#model.evaluate(test_set)
#model.save('./captioning_model')
#model = keras.models.load_model('./captioning_model', custom_objects={'masking': ConstantMask})

"""# Image caption generation"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib

visual_set = test_set.unbatch().shuffle(100).take(20)
for sample in visual_set:
    input_1 = tf.reshape(sample[0]['input_1'], (1,img_height,img_width,img_channels))
    input_2 = tf.reshape(tf.fill(tf.shape(sample[0]['input_2']), 0), (1,maxlen-1))
    
    i = 0
    found_stop_word = False
    caption = []
    while i < maxlen-1 and not found_stop_word:
        output = model.call([input_1, input_2])
        
        token_i = tf.argmax(output[0,i,:]).numpy()
        input_2_mod = input_2.numpy()
        input_2_mod[0,i] = token_i
        input_2 = tf.convert_to_tensor(input_2_mod)
        
        found_stop_word = token_i == stop_word_idx
        caption.append(token_i)
        
        i += 1

    print('Final sequence: ' + str(t.sequences_to_texts([caption])) + '\n')
    plt.imshow(input_1.numpy().astype(int)[0,:,:,:])
    plt.show()


