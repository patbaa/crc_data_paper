################################################################################
# imports

import sys
sys.path.remove('/home/pataki/.local/lib/python3.6/site-packages')

import os
import json
import ctypes
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from pathlib import Path
from network import get_model
from dataloader import DataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)

################################################################################
# parsing args

parser = argparse.ArgumentParser()
parser.add_argument("--train_meta_file", type=Path, help="Path of the  TSV \
                    file containing the filenames and labels", required=True)

parser.add_argument("--test_meta_file", type=Path, help="Path of the  TSV \
                    file containing the filenames and labels", required=True)

parser.add_argument("--model", type=str, default='ResNet50',
                    help="CNN model to use")

parser.add_argument("--GPU_ID", type=int, help="GPU ID to use", required=True)

parser.add_argument("--log_dir", type=Path, help="Directory for logs", 
                    required=True)

parser.add_argument("--epoch_frac", type=int, default=1, 
                    help="Divident for step per epoch")

parser.add_argument("--epochs", type=int, help="Number of epochs", default=10)

parser.add_argument('--lr', type=list, help='Learning rates for each epoch', 
                    default=[1e-3]*5 + [1e-4]*5)

parser.add_argument('--batch_size', type=list, help='Batch size', 
                    default=16)

parser.add_argument('--opt',type=str, help='Optimizer', default='sgd')

args = parser.parse_args()

################################################################################
# set GPU

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.GPU_ID)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.visible_device_list = str(args.GPU_ID)
set_session(tf.compat.v1.Session(config=config))

################################################################################
# label options

labels = ['lowgrade_dysplasia', 'highgrade_dysplasia', 'adenocarcinoma', 
          'suspicious_for_invasion', 'inflammation', 'resection_edge', 
          'tumor_necrosis', 'lymphovascular_invasion', 'artifact', 'normal']

################################################################################
# saving logs

args.log_dir.mkdir(parents=True, exist_ok=True)

with open(args.log_dir.joinpath('logs.txt'), 'w') as f:
    f.write('\n\n' + '#' * 80 + '\n\n')
    f.write(str(args.__dict__))
    f.write('\n\n' + '#' * 80 + '\n\n')

################################################################################
# model specific settings

if args.model == 'ResNet50':
    model = get_model('ResNet50', len(labels))
    preprocess_fn = resnet50_preprocess
    
if args.opt == 'sgd':
    opt = SGD(lr = args.lr[0])
elif args.opt == 'adam':
    opt = Adam(lr = args.lr[0])
    
################################################################################
# dataloader settings

trainDF = pd.read_csv(args.train_meta_file)
testDF  = pd.read_csv(args.test_meta_file)

train_gen = DataGenerator(fnames=trainDF.fname.values, 
                          labels=trainDF[labels].values, 
                          batch_size=args.batch_size, 
                          preprocess_input=preprocess_fn)

valid_gen = DataGenerator(fnames=testDF.fname.values, 
                          labels=testDF[labels].values, 
                          batch_size=args.batch_size, 
                          preprocess_input=preprocess_fn, shuffle=False)

################################################################################
# fitting the model

# accuracy where all the labels are the same for a multi label task
def correct_accuracy(y_true, y_pred, threshold=0.5):
    if threshold != 0.5:
        threshold = K.cast(threshold, y_pred.dtype)
        y_pred = K.cast(y_pred > threshold, y_pred.dtype)
    return K.cast(K.all(K.equal(y_true, K.round(y_pred)), axis=-1), K.floatx())

def lr_scheduler(epoch):
    return args.lr[epoch]

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

model.compile(optimizer=opt, loss='binary_crossentropy', 
              metrics=['binary_accuracy', 'categorical_accuracy', 
                       correct_accuracy])

history = model.fit(train_gen, validation_data=valid_gen,
                    epochs=args.epochs, workers=4, verbose=2,
                    steps_per_epoch = len(train_gen)//args.epoch_frac,
                    validation_steps = len(valid_gen),
                    callbacks=[lr_callback]
                    )

with open(args.log_dir.joinpath('logs.txt'), 'a') as f:
    f.write('\n\n' + '#' * 80 + '\n\n')
    f.write(str(history.history))

model.save(args.log_dir.joinpath('model.h5'))

################################################################################
# saving validation predictions

valid_gen = DataGenerator(fnames=testDF.fname.values, 
                          labels=testDF[labels].values, 
                          batch_size=1, 
                          preprocess_input=preprocess_fn, shuffle=False)

predictions = model.predict_generator(valid_gen)
for idx, i in enumerate(labels):
    testDF['pred_' + i] = predictions[:,idx] 
testDF.to_csv(args.log_dir.joinpath('preds.csv'))

################################################################################