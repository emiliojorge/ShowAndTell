import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from tensorflow import keras
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model

"""# Miscellaneous"""

img_height = 224
img_width = 224

test_fraction = 0.2

parser = argparse.ArgumentParser()
parser.add_argument('--dropout', type=float, help='recurrent dropout', default =0.0)
parser.add_argument('--lr', type=float, help='learning rate', default =0.01)
parser.add_argument('--cnn_top', type=str, help="end of CNN", default = "dense")
args = parser.parse_args()

print(args)
lr = args.lr
dropout = args.dropout
cnn_top = args.cnn_top


"""# Preprocessing of captions"""




df = pd.read_csv('train/captions.txt')
df = df[df.caption.notnull()]
captions = df[['caption']].to_numpy().flatten()
stop_word = 'STOPWORD'
captions = [x + ' ' + stop_word for x in captions]
#print(captions)


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

if cnn_top=="globalpool":
    cnn_out = keras.layers.GlobalMaxPooling2D()(output_vgg16_conv)

elif cnn_top=="dense":
    cnn_flatten = keras.layers.Flatten()(output_vgg16_conv)
    cnn_dense = keras.layers.Dense(512)(cnn_flatten)
    cnn_out = keras.layers.Dense(512)(cnn_dense)
else:
    raise NotImplementedError

cnn_seq = keras.layers.Reshape(target_shape=(1,512))(cnn_out)
cnn_seq = keras.layers.BatchNormalization()(cnn_seq)

class ConstantMask(keras.layers.Layer):
    def call(self, inputs):
        return inputs
    
    def compute_mask(self, inputs, mask=None):
        return tf.fill(tf.shape(inputs[:,:,0]), True)

# Turn output of VGG16 into sequence of 1 time step
cnn_seq_mask = ConstantMask()(cnn_seq)


# RNN input from training data (from tokenizer)
caption_input = keras.layers.Input(shape=(maxlen-1))
rnn_embed = keras.layers.Embedding(input_dim=num_words, output_dim=512, mask_zero=True)(caption_input)

# Concatenate CNN output and embedding output
merged = keras.layers.Concatenate(axis=1)([cnn_seq_mask, rnn_embed])

# LSTM layer with merged sequence as input
# For dropout, see [34], it is unclear if the parameter 'dropout' was used in the original paper
# Not sure how "ensembling" is used...


lstm = keras.layers.LSTM(512, recurrent_dropout = dropout, return_sequences=True)(merged)
lstm_output = keras.layers.Dense(num_words, activation='softmax')(lstm)
#cropping = tf.keras.layers.Cropping1D(cropping=(0,1))(lstm_output)
model = Model(inputs=[img_input, caption_input], outputs=lstm_output)
opt = tf.keras.optimizers.SGD(learning_rate=lr/batch_size,
                              momentum=0.0, 
                              nesterov=False, 
                              name="SGD") # See section 4.3.1

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM),
              optimizer=opt, 
              metrics=['accuracy'])




model.summary()

#model.load_weights("model.h5")
from datetime import datetime

file_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S_") + str(lr) + "_"+str(dropout) + "_" + cnn_top



cp_callback = keras.callbacks.ModelCheckpoint(
    filepath='./captioning_model/check_point/'+ file_stamp + 'weights_epoch_{epoch:02d}.hdf5',
    verbose=1, 
    save_weights_only=True,
    save_freq= 'epoch')


model.fit(train_set, epochs=20, callbacks=[cp_callback])
#model.evaluate(test_set)

"""# Image caption generation"""

print('Saving ./captioning_model/'+file_stamp+'/')
model.save('./captioning_model/'+file_stamp+'/')
model.save('./captioning_model/'+file_stamp+'/model.h5')

