import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import keras.backend as K
from keras.layers import Flatten, Dropout, LeakyReLU, Input, Activation, Dense, BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimuktaneous, normal_latent_sampling
from image_utils import dim_ordering_fix, dim_ordering_input, dim_ordering_reshape, dim_ordering_unfix

#This line allows mpl to run with no DISPLAY defind
mpl.use("Agg")

def model_generartor():
    nch = 256
    g_input = Input(shape=[100])
    H = Dense(nch * 14 * 14)(g_input)
    H = BatchNormalization()(H)
    H = Activation("relu")(H)
    H = dim_ordering_reshape(nch, 14)(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Conv2D(int(nch / 2), (3, 3), padding="same")(H)
    H = BatchNormalization()(H)
    H = Activation("relu")(H)
    H = Conv2D(int(nch / 4), (3, 3), padding="same")(H)
    H = BatchNormalization()(H)
    H = Activation("relu")(H)
    H = Conv2D(1, (1, 1), padding="same")(H)
    g_V = Activation("sigmoid")(H)
    return Model(g_input, g_V)
