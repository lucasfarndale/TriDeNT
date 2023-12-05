#################################################################################
# Kernel and Imports
#################################################################################

print("kernel working")
import os, sys

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf

from VICReg.vicreg_utils  import create_projector, create_resnet, create_adam_opt, save_vicreg_weights, load_vicreg_weights
from VICReg.dataset_utils import preprocess_ds, load_datasets
from VICReg.augmentations import *
from VICReg.analysis_utils import *
from VICReg.warmupcosine import WarmUpCosine

from ssl_models import VICReg

# Options for the dataset CPU utilisation
options = tf.data.Options()
options.threading.private_threadpool_size = 10

strategy = tf.distribute.MirroredStrategy()

#################################################################################
# Hyperparameters
#################################################################################

AUTO = tf.data.AUTOTUNE
SEED = 42

BATCH_SIZE  = 256
EPOCHS      = 100
IM_SIZE = 256
NUM_CHANNELS = 3
input_shape = (IM_SIZE,IM_SIZE,NUM_CHANNELS)

SHUFFLE_BUFFER = 2**10

# MAKE SURE THIS IS ALWAYS FALSE - REDUNDANT CODE FROM RELATED PROJECT
SPLIT_REP_FLAG = False

DATASET_SIZE = 100000
WARMUP_FRACTION = 0.1
STEPS_PER_EPOCH = DATASET_SIZE//BATCH_SIZE
WARMUP_EPOCHS = EPOCHS * WARMUP_FRACTION
WARMUP_STEPS = int(WARMUP_EPOCHS * STEPS_PER_EPOCH)

lr_decayed_fn = WarmUpCosine(learning_rate_base=1e-4,
                             total_steps=EPOCHS*STEPS_PER_EPOCH,
                             warmup_learning_rate=0.0,
                             warmup_steps=WARMUP_STEPS
                             )

SAVE_PATH = "."

# Path to .tfrecords file or see note below
record_path_train = "."
record_path_valid = "."

#################################################################################
# Dataset
#################################################################################
# I wrote this using binary files for efficiency but if you have a folder with
# images to load, just use 
# 
# image_ds = tf.keras.utils.image_dataset_from_directory(<directory_path>, 
#                                                        labels=None,
#                                                        label_mode=None,
#                                                        batch_size=None,
#                                                        image_size = (<image_size>,<image_size>),
#                                                       )
# 
# and modify 
# 
# train_ds = train_ds.map(lambda x: (augment_im(tf.transpose(x, [1,2,0])), augment_im(tf.transpose(x, [1,2,0]))))
# valid_ds = valid_ds.map(lambda x: (augment_im(tf.transpose(x, [1,2,0])), augment_im(tf.transpose(x, [1,2,0])))).
#
# If your images are channels last already, remove the transpose function.

augment_im = lambda x: custom_augment_multiplex(x, input_shape=(IM_SIZE,IM_SIZE,NUM_CHANNELS), output_shape=(IM_SIZE,IM_SIZE,NUM_CHANNELS))

feature_dict = feature_dict = {'he': tf.float32,
                               'ihc': tf.float32
                              }
(train_ds, valid_ds) = load_datasets([record_path_train,record_path_valid], output_dict=feature_dict)
train_ds = train_ds.map(lambda x: (augment_im(x['he']), augment_im(x['he']), augment_im(x['ihc'])), num_parallel_calls=AUTO)
valid_ds = valid_ds.map(lambda x: (augment_im(x['he']), augment_im(x['he']), augment_im(x['ihc'])), num_parallel_calls=AUTO)
train_ds = preprocess_ds(train_ds, batch_size=BATCH_SIZE, seed=SEED, pre=AUTO, shuffle_no=SHUFFLE_BUFFER, rei=True).with_options(options)
valid_ds = valid_ds.batch(BATCH_SIZE).prefetch(AUTO).with_options(options)

#################################################################################
# Model & Training
#################################################################################
with strategy.scope():
    encoder1        = create_resnet(input_shape)
    projector1      = create_projector()
    encoder2        = create_resnet(input_shape)
    projector2      = create_projector()
    
    enc_list            = [encoder1, encoder2]
    proj_list           = [projector1, encoder2]

    encoder_indices = [0,0,1]
    projector_indices = [0,0,1]
    optimiser = create_adam_opt(lr_decayed_fn)

    vicreg = VICReg(encoder_list=enc_list, projector_list=proj_list, encoder_indices=encoder_indices, projector_indices=projector_indices, split_rep=SPLIT_REP_FLAG)
    vicreg.compile(optimizer=optimiser)
    vicreg.fit(train_ds,
               epochs=EPOCHS,
               validation_data=valid_ds
              )
save_vicreg_weights(SAVE_PATH, enc_list, proj_list)

if __name__=="__main__":
    main()
