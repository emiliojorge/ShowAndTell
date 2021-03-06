
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

""" ------------------ Tokenization -------------------- """

t = Tokenizer()
t.fit_on_texts(captions)

num_word_count_over_5 = 0
removed_tokens = set()
filtered_vocab = dict()
words_from_filtered = dict()
for item in t.word_counts.items():
    token = t.texts_to_sequences([[item[0]]])[0][0]
    if item[1] >= 5:
        num_word_count_over_5+=1
        filtered_vocab[token] = num_word_count_over_5
        words_from_filtered[num_word_count_over_5] = item[0]
    else:
        removed_tokens.add(token)

orig_seq = t.texts_to_sequences(captions)
filtered_seq = []
for sentence in orig_seq:
    filtered_seq.append([])
    for token in sentence:
        if token not in removed_tokens:
            filtered_seq[-1].append(token)


num_words = num_word_count_over_5 + 1
seq = [[filtered_vocab[y] for y in x] for x in filtered_seq]

# Maximum length of any sequence, plus one stop word
maxlen = max([len(x) for x in seq])

padded_captions = pad_sequences(seq, maxlen=maxlen, padding='post')
#print(padded_captions)

stop_word_idx = filtered_vocab[t.texts_to_sequences([stop_word])[0][0]]


""" ------------------  Preprocessing of images -------------------- """

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
dataset = dataset.map(lambda x, y, z: (tf.image.decode_jpeg(x, channels=3), y, z))
dataset = dataset.map(lambda x, y, z: (tf.image.resize(x,[img_height, img_width]), y, z))
dataset = dataset.map(lambda x, y, z: ({'input_1':x, 'input_2':y}, z))
#dataset = dataset.batch(32, drop_remainder=True)
n_data = padded_captions.shape[0]

batch_size = 32
test_set = dataset.take(int(test_fraction*n_data)).shuffle(1000).batch(batch_size)
train_set = dataset.skip(int(test_fraction*n_data)).shuffle(1000).batch(batch_size)

"""# Model"""



"""# Load pre-trained VGG16 model"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model

# VGG16
vgg16_conv = vgg16.VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

img_input = keras.layers.Input(shape=(224,224,3))
vgg16_conv.trainable = False

# We only change the activation of the last layer from softmax to relu
vgg16_conv.layers[-1].activation = keras.activations.relu
vgg16_conv.compile(loss="mse") 
#vgg16_conv.summary()
#print(vgg16_conv.get_config())
output_vgg16_conv = vgg16_conv(img_input)

# Turn output of VGG16 into sequence of 1 time step

# If 'include_top' is False
cnn_flatten = keras.layers.Flatten()(output_vgg16_conv)
cnn_dense = keras.layers.Dense(512)(cnn_flatten)
cnn_dense = keras.layers.Dense(512)(cnn_dense)
cnn_seq = keras.layers.Reshape(target_shape=(1,512))(cnn_dense)
cnn_seq = keras.layers.BatchNormalization()(cnn_seq)

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
lstm = keras.layers.LSTM(512, recurrent_dropout = 0.0, return_sequences=True)(merged)
lstm_output = keras.layers.Dense(num_words, activation='softmax')(lstm)
#cropping = tf.keras.layers.Cropping1D(cropping=(0,1))(lstm_output)
model = Model(inputs=[img_input, caption_input], outputs=lstm_output)
opt = tf.keras.optimizers.SGD(learning_rate=0.01/batch_size,
                              momentum=0.0, 
                              nesterov=False, 
                              name="SGD") # See section 4.3.1

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM),
              optimizer=opt, 
              metrics=['accuracy'])
model.summary()

model.load_weights('./captioning_model/2020-11-18-10:53:02/model.h5')
#model.fit(train_set, epochs=20)
#model.evaluate(test_set)
#model.save('./captioning_model')
#model = keras.models.load_model('./captioning_model', custom_objects={'masking': ConstantMask})

"""# Image caption generation"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    return v/norm

#visual_set = train_set.unbatch().shuffle(100).take(20)
visual_set = test_set.unbatch().shuffle(100).take(20)
outputs = []
for sample in visual_set:
    input_1 = tf.reshape(sample[0]['input_1'], (1,img_height,img_width,img_channels))
    input_2 = tf.reshape(tf.fill(tf.shape(sample[0]['input_2']), 0), (1,maxlen-1))
    
    i = 0
    found_stop_word = False
    caption = []
    #while i < 1 and not found_stop_word:
    while i < maxlen-1 and not found_stop_word:
        output = model.call([input_1, input_2])
        
        token_i = tf.argmax(output[0,i,:]).numpy()
        #token_i = np.argmax(np.random.multinomial(1, pvals=normalize(output[0,i,:].numpy())))
        input_2_mod = input_2.numpy()
        input_2_mod[0,i] = token_i
        input_2 = tf.convert_to_tensor(input_2_mod)
        
        found_stop_word = token_i == stop_word_idx or token_i == 0
        if not found_stop_word:
            caption.append(token_i)
        
        i += 1

    output_array = output[0,i,:].numpy()
    outputs.append(output_array.reshape((1, len(output_array))))                              
    translated_caption = [words_from_filtered[y] for y in caption]
    print('Final sequence: ' + ' '.join(translated_caption) + '\n')
    fig, ax1 = plt.subplots(1,1)
    ax1.set_title(' '.join(translated_caption))
    ax1.imshow(input_1.numpy().astype(int)[0,:,:,:])
    plt.show()
outputs_array = np.vstack(outputs)

# Testing ouput of layers
layer_test = keras.backend.function([img_input, caption_input],[merged])
layer_outputs = []
for sample in visual_set:
    input_1 = tf.reshape(sample[0]['input_1'], (1,img_height,img_width,img_channels))
    input_2 = tf.reshape(tf.fill(tf.shape(sample[0]['input_2']), np.random.randint(1,100)), (1,maxlen-1))
    #input_2 = tf.reshape(tf.fill(tf.shape(sample[0]['input_2']), 0), (1,maxlen-1))
    layer_output = layer_test([input_1, input_2])[0][0,0,:].reshape(1,512)
    print('Norm from CNN output: ' + str(np.linalg.norm(layer_test([input_1, input_2])[0][0,0,:].reshape(1,512), ord=2)))
    print('Norm from embedding output: ' + str(np.linalg.norm(layer_test([input_1, input_2])[0][0,1,:].reshape(1,512), ord=2)))
    layer_outputs.append(layer_output)
layer_outputs_array = np.vstack(layer_outputs)