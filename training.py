from keras import backend as K
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint
from Image_Generator import TextImageGenerator
from Model import get_Model
from parameter import *
import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.tensorflow_backend.set_session(sess)

K.set_learning_phase(0)

# # Model description and training

model = get_Model(training=True)
model.summary()

try:
    model.load_weights('LSTM+BN4--26--0.011.hdf5')
    print("...Previous weight data...")
except:
    print("...New weight data...")
    pass

train_file_path = './DB/train/'
tiger_train = TextImageGenerator(train_file_path, img_w, img_h, batch_size, downsample_factor)
tiger_train.build_data()

valid_file_path = './DB/test/'
tiger_val = TextImageGenerator(valid_file_path, img_w, img_h, val_batch_size, downsample_factor)
tiger_val.build_data()

ada = Adadelta()

def out(y_true, y_pred):
    return tf.reduce_mean(y_pred)
def cer(y_true, y_pred):
    return tf.reduce_mean(y_pred)
def ser(y_true, y_pred):
    return tf.reduce_mean(tf.to_float(y_pred!=0))


early_stop = EarlyStopping(monitor='val_lev_cer', min_delta=0.001, patience=4, mode='min', verbose=1)
checkpoint = ModelCheckpoint(filepath='LSTM+BN5--{epoch:02d}--{val_lev_cer:.3f}.hdf5', monitor='val_lev_cer', verbose=1, mode='min', save_best_only=True)
# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': out, 'lev': out}, metrics={'lev':[cer, ser]}, loss_weights=[1.0, 0.0], optimizer=ada)

# captures output of softmax so we can decode the output during visualization
model.fit_generator(generator=tiger_train.next_batch(),
                    steps_per_epoch=int(tiger_train.n / batch_size),
                    epochs=30,
                    callbacks=[checkpoint],
                    validation_data=tiger_val.next_batch(),
                    validation_steps=int(tiger_val.n / val_batch_size),
                    verbose=2)
