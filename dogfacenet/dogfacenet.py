"""
DogFaceNet
The main DogFaceNet implementation
This file contains:
 - Data loading
 - Model definition
 - Model training

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from online_training import *

#----------------------------------------------------------------------------
# Config.

PATH        = '../../after_4_bis/' # Path to the directory of the saved dataset
os.mkdir('../output') #Create a directory with Name 'output' followed by history and model directory inside output.
os.mkdir('../output/history/')
os.mkdir('../output/model/')
PATH_SAVE   = '../output/history/'                      # Path to the directory where the history will be stored
PATH_MODEL  = '../output/model/'             # Path to the directory where the model will be stored
SIZE        = (224,224,3)                               # Size of the input images
TEST_SPLIT  = 0.1                                       # Train/test ratio

LOAD_NET    = False                                     # Load a network from a saved model? If True NET_NAME and START_EPOCH have to be precised
NET_NAME    = 'dogfacenet'                   # Network saved name
START_EPOCH = 0                                         # Start the training at a specified epoch
NBOF_EPOCHS = 10                                      # Number of epoch to train the network
HIGH_LEVEL  = True                                      # Use high level training ('fit' keras method)
STEPS_PER_EPOCH = 300                                   # Number of steps per epoch
VALIDATION_STEPS = 30                                   # Number of steps per validation

#----------------------------------------------------------------------------
# Import the dataset.

assert os.path.isdir(PATH), '[Error] Provided PATH for dataset does not exist.'

print('Loading the dataset...')

filenames = np.empty(0)
labels = np.empty(0)
idx = 0
for root,dirs,files in os.walk(PATH): #Iterate through all the directories and Load the images names as well as labels
    if len(files)>1:
        for i in range(len(files)):
            files[i] = root + '/' + files[i]
        filenames = np.append(filenames,files) #appending file names
        labels = np.append(labels,np.ones(len(files))*idx) #appending labels
        idx += 1
assert len(labels)!=0, '[Error] No data provided.' #if dataset does not exists

print('Done.')

print('Total number of imported pictures: {:d}'.format(len(labels))) #Total Number of Pictures

nbof_classes = len(np.unique(labels))
print('Total number of classes: {:d}'.format(nbof_classes))

#----------------------------------------------------------------------------
# Split the dataset 10% for Testing other for training.

nbof_test = int(TEST_SPLIT*nbof_classes)

keep_test = np.less(labels,nbof_test)
keep_train = np.logical_not(keep_test)

filenames_test = filenames[keep_test]
labels_test = labels[keep_test]

filenames_train = filenames[keep_train]
labels_train = labels[keep_train]

print("Number of training data: " + str(len(filenames_train)))
print("Number of training classes: " + str(nbof_classes-nbof_test))
print("Number of testing data: " + str(len(filenames_test)))
print("Number of testing classes: " + str(nbof_test))

#----------------------------------------------------------------------------
# Loss definition.

alpha = 0.3 #Regularization Parameter of Triplet loss
def triplet(y_true,y_pred):
    '''
    Triplet loss is used for one shot learning. (Few examples of each class used in training.) 
    A CNN network is used to get embedding vector of images , then Anchor Image is compared with two image samples (Pos and Negative).
    Distance between Anchor and Pos/Neg is calculated
    Sum of taken for Anchor_Postive , Achor_Negative and Aplha. 
    '''
    a = y_pred[0::3] #Anchor
    p = y_pred[1::3] #Positive
    n = y_pred[2::3] #Nagative
    
    ap = K.sum(K.square(a-p),-1) #Distance between Anchor and Positive
    an = K.sum(K.square(a-n),-1) #Distance between Anchor and Negative
  
    return K.sum(tf.nn.relu(ap - an + alpha)) #Retuurn relu function over tripplet loss

def triplet_acc(y_true,y_pred):
    #Accuracy Computation for Triplet loss
    a = y_pred[0::3] #Getting Anchor for comperision
    p = y_pred[1::3] #Getting Positive
    n = y_pred[2::3] #Getting Negative
    
    ap = K.sum(K.square(a-p),-1) #SUm of Squre between Anchor and Positve
    an = K.sum(K.square(a-n),-1) #SUm of Squre between Anchor and Negative
    
    return K.less(ap+alpha,an) #Min (Element wise) betweeen AchorPostive and Negative is taken

#----------------------------------------------------------------------------
# Model definition.

if LOAD_NET:
    
    print('Loading model from {:s}{:s}.{:d}.h5 ...'.format(PATH_MODEL,NET_NAME,START_EPOCH))
    #Model loading if made
    model = tf.keras.models.load_model(
        '{:s}{:s}.{:d}.h5'.format(PATH_MODEL,NET_NAME,START_EPOCH),
        custom_objects={'triplet':triplet,'triplet_acc':triplet_acc})
    
    print('Done.')
else:
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, DepthwiseConv2D
    from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Lambda, BatchNormalization

    """
    Model number 12: Paper version: a modified ResNet with Dropout layers and without bottleneck layers
    """
    #Model is defined here
    print('Defining model {:s} ...'.format(NET_NAME))
    #Emdeding Vector Size
    emb_size = 32
    #input layer 
    inputs = Input(shape=SIZE)
    #Convolutional Layers
    x = Conv2D(16, (7, 7), (2, 2), use_bias=False, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x) #Batch Normalization for normalization of different batches
    x = MaxPooling2D((3,3))(x)
    #ResNet style Model Structure
    for layer in [16,32,64,128,512]:

        x = Conv2D(layer, (3, 3), strides=(2,2), use_bias=False, activation='relu', padding='same')(x)
        r = BatchNormalization()(x)
        
        x = Conv2D(layer, (3, 3), use_bias=False, activation='relu', padding='same')(r)
        x = BatchNormalization()(x)
        r = Add()([r,x])
        
        x = Conv2D(layer, (3, 3), use_bias=False, activation='relu', padding='same')(r)
        x = BatchNormalization()(x)
        x = Add()([r,x])
        
    #Global Average Pooling for making it vector like
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x) #Flatten the vectors
    x = Dropout(0.5)(x)
    x = Dense(emb_size, use_bias=False)(x)
    outputs = Lambda(lambda x: tf.nn.l2_normalize(x,axis=-1))(x)

    model = tf.keras.Model(inputs,outputs)

    model.compile(loss=triplet,
                optimizer='adam',
                metrics=[triplet_acc])

    print('Done.')

print(model.summary())

#----------------------------------------------------------------------------
# Model training.

if HIGH_LEVEL:
    """
    Hard training: high level of implementation for fine tuning
    """

    histories = []
    crt_loss = 0.6 
    crt_acc = 0 #intialize accuracy
    batch_size = 3*10 #30 is size of batch
    nbof_subclasses = 40 

    # Create saving folders
    if not os.path.isdir(PATH_MODEL):
        os.makedirs(PATH_MODEL)
    if not os.path.isdir(PATH_SAVE):
        os.makedirs(PATH_SAVE)

    # Bug fixed: keras models are to be initialized by a training on a single batch
    for images_batch,labels_batch in online_adaptive_hard_image_generator(
        filenames_train,
        labels_train,
        model,
        crt_acc,
        batch_size,
        nbof_subclasses=nbof_subclasses):
        h = model.train_on_batch(images_batch,labels_batch)
        break


    for i in range(START_EPOCH,START_EPOCH+NBOF_EPOCHS):
        print("Beginning epoch number: "+str(i))

        hard_triplet_ratio = np.exp(-crt_loss * 10 / batch_size)
        nbof_hard_triplets = int(batch_size//3 * hard_triplet_ratio)
        
        print("Current hard triplet ratio: " + str(hard_triplet_ratio))
        
        histories += [model.fit_generator(
            online_adaptive_hard_image_generator(filenames_train,labels_train,model,crt_loss,batch_size,nbof_subclasses=nbof_subclasses),
            steps_per_epoch=STEPS_PER_EPOCH,
            epochs=1,
            validation_data=image_generator(filenames_test,labels_test,batch_size,use_aug=False),
            validation_steps=VALIDATION_STEPS)] #model training and saving record in hisotry
        
        crt_loss = histories[-1].history['loss'][0]
        crt_acc = histories[-1].history['triplet_acc'][0]

        # Save model
        model.save('{:s}{:s}.{:d}.h5'.format(PATH_MODEL,NET_NAME,i))
        
        # Save history
        loss = np.empty(0)
        val_loss = np.empty(0)
        acc = np.empty(0)
        val_acc = np.empty(0)

        for history in histories:
            loss = np.append(loss,history.history['loss'])
            val_loss = np.append(val_loss,history.history['val_loss'])
            acc = np.append(acc,history.history['triplet_acc'])
            val_acc = np.append(val_acc,history.history['val_triplet_acc'])

        history_ = np.array([loss,val_loss,acc,val_acc])
        np.save('{:s}{:s}.{:d}.npy'.format(PATH_SAVE,NET_NAME,i),history_)

else:
    """
    Training: lower level of implementation
    """

    max_epoch = NBOF_EPOCHS + START_EPOCH

    max_step = 300 
    max_step_test = 30
    batch_size = 3*10


    tot_loss_test = 0
    mean_loss_test = 0

    tot_acc_test = 0
    mean_acc_test = 0

    # Save
    loss = []
    val_loss = []
    acc = []
    val_acc = []

    for epoch in range(START_EPOCH,max_epoch):
        
        step = 1
        
        tot_loss = 0
        mean_loss = 0

        tot_acc = 0
        mean_acc = 0
        
        # Training
        for images_batch,labels_batch in online_adaptive_hard_image_generator(
            filenames_train,
            labels_train,
            model,
            mean_acc,
            batch_size,
            nbof_subclasses=10
            ):


            h = model.train_on_batch(images_batch,labels_batch)
            tot_loss += h[0]
            mean_loss = tot_loss/step
            tot_acc += h[1]
            mean_acc = tot_acc/step
            #clear_output()

            # hard_triplet_ratio = np.exp(-mean_loss * 10 / batch_size)
            hard_triplet_ratio = max(0,1.2/(1+np.exp(-10*mean_acc+5.3))-0.19)

            print(
                "Epoch: " + str(epoch) + "/" + str(max_epoch) +
                ", step: " + str(step) + "/" + str(max_step) + 
                ", loss: " + str(mean_loss) + 
                ", acc: " + str(mean_acc) +
                ", hard_ratio: " + str(hard_triplet_ratio)
            )
            print(
                "Test loss: " + str(mean_loss_test) + 
                ", test acc: " + str(mean_acc_test)
            )

            if step == max_step:
                break
            step+=1
        
        loss += [mean_loss]
        acc += [mean_acc]
        
        # Testing
        step = 1
        
        tot_loss_test = 0
        mean_loss_test = 0

        tot_acc_test = 0
        mean_acc_test = 0
        
        for images_batch,labels_batch in image_generator(filenames_test,labels_test,batch_size,use_aug=False):
            h = model.test_on_batch(images_batch,labels_batch)

            tot_loss_test += h[0]
            mean_loss_test = tot_loss_test/step
            tot_acc_test += h[1]
            mean_acc_test = tot_acc_test/step

            if step == max_step_test:
                break
            step+=1
        
        val_loss += [mean_loss_test]
        val_acc += [mean_acc_test]
        
        # Saving the fine tuned model
        model.save('{:s}{:s}.{:d}.h5'.format(PATH_MODEL,NET_NAME,epoch))
        history_ = np.array([loss,val_loss,acc,val_acc])
        np.save('{:s}{:s}.{:d}.npy'.format(PATH_SAVE,NET_NAME,epoch),history_)
