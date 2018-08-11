'''

vgg_simple.py is a CNN loosely based on VGG net.  It uses a couple of
convolutional blocks (3x3 filters with same padding, relu
activation, batch normalization, max pooling using 3x3 and 10% drop
out to avoud ovetfitting) followed by a single fully connected
layer with 10% dropout and batch normalization and relu activation,
followed by an output layer as big as the number of entities being
classified, this time with a sigmoid activation function.

It reads the h5.db file to grab the training, dev and test sets but
uses data augmentation to distort the images in the h5.db to
hopefully help with the relative scarcity of data.

For optimization, I use just Adam.
'''

import numpy as np
import numpy.random
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adadelta, Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import configparser
import h5py
from pprint import pprint
import time
import os.path
import os
    
INIT_LR = 0.001
EPOCHS = 100
BS=128
MODEL_INPUT_FILENAME = 'vggsimple_v1.model'
MODEL_OUTPUT_FILENAME = 'vggsimple_v1.model'
PLOT_FILENAME = MODEL_OUTPUT_FILENAME + ".png"

np.random.seed(1)
DB_FILE_NAME = 'h5.db'

datasets = dict()
total_records = 0
with h5py.File(DB_FILE_NAME, mode='r') as f:
    for group in f:
        total_records_in_group = 0
        print("Found group " + group)
        datasets[group] = dict()
        for dataset in f[group]:
            print("\tFound dataset " + dataset + " in group " + group)
            datasets[group][dataset] = f[group][dataset].value
            size = len(datasets[group][dataset])
            print("\t\tDataset " + dataset + " contains " + str(size) + " records")
            total_records_in_group += size
        # divide by three to avoid triple counting filenames, inputs and labels
        total_records_in_group = int(total_records_in_group/3)
        print("\t" + str(total_records_in_group) + " records were read in total from dataset")
        total_records += total_records_in_group
print(str(total_records) + " records were read across all datasets")

config = configparser.ConfigParser()
config.read('config.ini')
entities = config.get('general', 'entitiesToLookFor').split(',')

assert('training' in datasets)
assert('dev' in datasets)
assert('test' in datasets)
assert('inputs' in datasets['training'])
assert('outputs' in datasets['training'])
assert('inputs' in datasets['dev'])
assert('outputs' in datasets['dev'])
assert('inputs' in datasets['test'])
assert('outputs' in datasets['test'])

print("Input shape:")
print(datasets['training']['inputs'].shape)
print("Output shape:")
print(datasets['training']['outputs'].shape)

t1 = time.time()

model = None
print(os.getcwd())
print(os.path.isfile(MODEL_INPUT_FILENAME))

if (os.path.isfile(MODEL_INPUT_FILENAME)):
    print("Loading model from disk")
    model = load_model(MODEL_INPUT_FILENAME)
else:
    model = Sequential()

    chanDim = -1
    model.add(Conv2D(32, (3, 3), padding="same",
        input_shape=datasets['training']['inputs'][0].shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.1))

    # (CONV => RELU) * 2 => POOL
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    # (CONV => RELU) * 2 => POOL
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    # Output layer
    model.add(Dense(len(entities)))
    model.add(Activation('sigmoid'))
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

    try:
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    except Exception as e:
        print(e)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

earlystop = EarlyStopping(monitor='loss', min_delta=0.0000000001, patience=5, verbose=1, mode='auto')

H = model.fit_generator(
	aug.flow(datasets['training']['inputs'], datasets['training']['outputs'], batch_size=BS),
	validation_data=(datasets['dev']['inputs'], datasets['dev']['outputs']),
	steps_per_epoch=len(datasets['training']['inputs']) // BS,
	epochs=EPOCHS, verbose=1, callbacks=[earlystop])

scores  = model.evaluate(datasets['dev']['inputs'], datasets['dev']['outputs'], verbose=1)
print("Accuracy on dev set:")
print(scores[1])
scores  = model.evaluate(datasets['test']['inputs'], datasets['test']['outputs'], verbose=1)
print("Accuracy on test set:")
print(scores[1])


per_entity_stats = dict()
for entity in entities:
    # correctly predicted absent, correctly predicted present, incorrectly predicted absent, incorrectly predicted present
    per_entity_stats[entity] = [0, 0, 0, 0]
predictions = np.rint(model.predict(datasets['test']['inputs'])).astype(np.uint8)
for output, fname, predicted in zip(datasets['test']['outputs'], datasets['test']['filenames'], predictions):
    for i in range(len(entities)):
        classification = output[i]
        if (predicted[i] != classification):
            print("INCORRECT predicted was " + str(predicted[i]) + " classification is " + str(classification) + \
                " for entity " + str(entities[i]) + " - filename is " + fname)
            if(predicted[i] == 1):
                per_entity_stats[entities[i]][3] += 1
            else:
                per_entity_stats[entities[i]][2] += 1
        else:
            if(predicted[i] == 1):
                per_entity_stats[entities[i]][1] += 1
            else:
                per_entity_stats[entities[i]][0] += 1
            
        if (predicted[i] == 1 or classification == 1):
            print("++CORRECT predicted was " + str(predicted[i]) + " classification is " + str(classification) + \
                " for entity " + str(entities[i]) + " - filename is " + fname)

for entity in per_entity_stats:
    print (entity + " correctly predicted present:     " + str(per_entity_stats[entity][1]))
    print (entity + " correctly predicted absent:      " + str(per_entity_stats[entity][0]))
    print (entity + " incorrectly predicted present:   " + str(per_entity_stats[entity][3]))
    print (entity + " incorrectly predicted absent:    " + str(per_entity_stats[entity][2]))
    false_pos = 100 * per_entity_stats[entity][3] / (per_entity_stats[entity][3]+per_entity_stats[entity][1])
    true_pos = 100 * per_entity_stats[entity][1] / (per_entity_stats[entity][3]+per_entity_stats[entity][1])
    false_neg = 100 * per_entity_stats[entity][2] / (per_entity_stats[entity][0]+per_entity_stats[entity][2])
    true_neg = 100 * per_entity_stats[entity][0] / (per_entity_stats[entity][0]+per_entity_stats[entity][2])
    print (entity + " false positive rate:  " + str(false_pos))
    print (entity + " true positive rate:  " + str(true_pos))
    print (entity + " false negative rate:  " + str(false_neg))
    print (entity + " true negative rate:  " + str(true_neg))
    

print("\nModel construction, training and evaluation time is:")
print(time.time() -1)
# Save the model itself
model.save(MODEL_OUTPUT_FILENAME)
