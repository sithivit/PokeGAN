import os
from PIL import Image
import numpy as np
import random
from math import sqrt

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, Dropout, Input, BatchNormalization, \
    Concatenate, Reshape, Conv2DTranspose, LeakyReLU
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def load_data():
    cur_path = os.getcwd()
    classes = [x[0] for x in os.walk(cur_path)][1:15]
    num_len = len(classes)
    data = []
    labels = []
    for i in classes:
        try:
            print(i.split('\\')[-1])
            images = os.listdir(i)
            for a in images:
                if not a.endswith('svg'):
                    image = Image.open(i + '\\' + a)
                    try:
                        image = image.resize((60, 60))
                        image = np.array(image)
                        image = image.reshape((60, 60, 3))

                        data.append(image)
                        labels.append(i.split('\\')[-1])
                    except:
                        pass
        except:
            pass

    return data, labels, num_len + 2


data, labels, num_len = load_data()

data = np.array(data)

num_to_labels = os.listdir(os.getcwd())
labels = [num_to_labels.index(i) for i in labels]
labels = np.array(labels)

print(data.shape, labels.shape)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

print(x_train.shape, x_test.shape)

y_train = to_categorical(y_train, num_len)
y_test = to_categorical(y_test, num_len)
input_shape = x_train[1:]


def generator(latent_dim, code_dim, label):
    init = RandomNormal(stddev=0.02)
    in_lat = Input(shape=(latent_dim,))
    in_code = Input(shape=(code_dim,))
    label = Input(shape=(label,))
    n_nodes = 15 * 15 * 128

    in_latent_code = Concatenate()([in_lat, in_code, label])

    # Foundation for 15 x 15 images
    x = Dense(n_nodes, kernel_initializer=init)(in_latent_code)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Reshape((15, 15, 128))(x)

    # normal
    x = Conv2D(256, 5, padding='same', kernel_initializer=init)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # upsameple to 30x30
    x = Conv2DTranspose(128, 5, strides=(2, 2), padding='same', kernel_initializer=init)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # upsame pl to 60x60
    x = Conv2DTranspose(3, 5, strides=(2, 2), padding='same', kernel_initializer=init)(x)
    out_layer = Activation('tanh')(x)
    model = Model([in_lat, in_code, label], out_layer)
    return model


def discriminator(n_cat):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=(60, 60, 3))
    x = Conv2D(64, 4, strides=2, padding='same', kernel_initializer=init, input_shape=(60, 60, 3))(in_image)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(128, 4, strides=2, padding='same', kernel_initializer=init)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)

    out_classifier = Dense(1, activation='sigmoid')(x)
    d_model = Model(in_image, out_classifier)
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=2e-4))

    q = Dense(128)(x)
    q = BatchNormalization()(q)
    q = LeakyReLU(alpha=0.1)(q)
    out_codes = Dense(n_cat, activation='softmax')(q)

    q_model = Model(in_image, out_codes)
    return d_model, q_model


def gan(g_model, d_model, q_model):
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    d_output = d_model(g_model.output)
    q_output = q_model(g_model.output)
    model = Model(g_model.input, [d_output, q_output])

    model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=Adam(lr=2e-4))
    return model


def generate_latent_vector(latent_dim, code_dim, n_cat, n_samples):
    z_latent = np.random.randn(latent_dim * n_samples)
    z_latent = z_latent.reshape(n_samples, latent_dim)
    c_1 = tf.random.uniform([n_samples, code_dim], minval=-1, maxval=1, seed=42)
    cat_codes = np.random.randint(0, n_cat, n_samples)
    cat_codes = to_categorical(cat_codes, num_classes=n_cat)
    return [z_latent, c_1, cat_codes]


def load_real_samples(x_train):
    X = np.expand_dims(x_train, axis=-1)
    X = (X.astype('float32') - 127.5) / 127.5
    return X


def generate_real_samples(dataset, n_samples):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = tf.ones((n_samples, 1))
    return X, y


def generate_fake_samples(generator, latent_dim, code_dim, n_cat, n_samples):
    z_input, code, cat_code = generate_latent_vector(latent_dim, code_dim, n_cat, n_samples)
    print(z_input.shape, code.shape, cat_code.shape)
    images = generator.predict([z_input, code, cat_code])
    y = np.zeros((n_samples, 1))
    return images, y


def train(g_model, d_model, gan_model, dataset, latent_dim, n_code, n_cat, n_epochs=100, n_batch=64):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    n_steps = bat_per_epo + n_epochs

    half_batch = int(n_batch / 2)
    for i in range(n_steps):
        X_real, y_real = generate_real_samples(dataset, half_batch)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_code, n_cat, half_batch)
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)

        [z_input, code, cat_codes] = generate_latent_vector(latent_dim, n_code, n_cat, n_batch)
        y_gan = tf.ones((n_batch, 1))
        _, g_1, g_2 = gan_model.train_on_batch([z_input, code, cat_codes])


def create_plot(examples, n_examples):
    for i in range(n_examples):
        plt.subplot(sqrt(n_examples), sqrt(n_examples), 1 + i)

    plt.axis("off")
    plt.imshow(examples[i, :, :, 0], cmap='gray_r')
    plt.show()


n_cat = 152
latent_dim = 90
code_dim = 10

d_model, q_model = discriminator(n_cat)
g_model = generator(latent_dim, code_dim, n_cat)

gan_model = gan(g_model, d_model, q_model)

train(g_model, d_model, gan_model, x_train, latent_dim, code_dim, n_cat)

"""
z = Input(shape=(90,))
c = Input(shape=(10,))
x = Input(shape=input_shape)


def mi_loss(c, q_of_c_given_x):
    return -K.mean(K.sum(c * K.log(q_of_c_given_x + K.epsilon()), axis=1))


def generator(inputs):
    z, c = inputs
    x = Dense(128, activation='relu')(tf.concat([z, c], axis=-1))
    x = Dense(256, activation='relu')(x)
    output = Dense(50 * 50, activation='sigmoid')(x)
    return output


def discriminator(inputs):
    x, c = inputs
    x = Dense(256, activation='relu')(tf.concat([x, c], axis=-1))
    x = Dense(128, activation='relu')(x)
    logits = Dense(1)(x)
    output = Activation('sigmoid')(logits)
    return output, logits


# Generate synthetic data
g_z = generator([z, c])

# Discriminate real data
D_real, D_real_logits = discriminator([x, c])

D_fake, D_fake_logits = discriminator([g_z, c])

discriminator_model = Model([x, c], [D_real, D_real_logits])

generator_model = Model([z, c], g_z)

discriminator_model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=2e-4),
    matrics=['accuracy'],
)

generator_model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=2e-4),
    matrics=['accuracy'],
)

infoGAN = Model([z, c], )
"""
