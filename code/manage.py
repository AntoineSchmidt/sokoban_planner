import tensorflow as tf
from keras import backend as K

# Manage Keras session

def startSession(memory=0.3):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = memory
    K.set_session(tf.Session(config=config))

def finishSession():
    K.clear_session()
    K.get_session().close()