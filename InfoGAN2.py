import os

import numpy as np
import keras.layers
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.keras.utils import to_categorical

from PIL import Image

from sklearn.model_selection import train_test_split


def create_generator_continuous(n_filter=128, input_size=224):
    input = keras.layers.Input(shape=(input_size,))

    x = keras.layers.Dense(units=1024, use_bias=False)(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)

    x = keras.layers.Dense(units=15 * 15 * 128, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)

    reshape = keras.layers.Reshape(taget_shape=(15, 15, 128))(x)

    nf = n_filter

    x = keras.layers.Conv2DTranspose(nf, kernel_size=(4, 4), stides=(2, 2), padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)

    # Number of filters halved after each transposed convolutional layer
    nf = nf // 2

    x = keras.layers.Conv2DTranspose(nf, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)

    output = keras.layers.Conv2DTranspose(3, kernel_size=(4, 4), strides=(1, 1), padding='same', activation='tanh')(x)
    model = keras.models.Model(inputs=input, outputs=output)
    return model


def create_discriminator_continuous(n_filters=64, n_class=152, input_shape=(60, 60, 3)):
    image_input = keras.layers.Input(shape=input_shape)

    nf = n_filters
    x = keras.layers.Conv2D(nf, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=True)(image_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)

    nf = nf * 2

    x = keras.layers.Conv2D(nf, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)

    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(1024, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)

    d_output = keras.layers.Dense(1, activation='sigmoid')(x)

    # Auxiliary output

    q = keras.layers.Dense(128, use_bias=False)(x)
    q = keras.layers.BatchNormalization()(q)
    q = keras.layers.LeakyReLU(alpha=0.1)(q)

    clf_out = keras.layers.Dense(n_class, activation='softmax')(q)

    mu = keras.layers.Dense(1)(q)
    sigma = keras.layers.Dense(1, activation=lambda y: tf.math.exp(y))(q)

    d_model = keras.models.Model(inputs=image_input, outputs=d_output)

    q_model = keras.models.Model(inputs=image_input, outputs=[clf_out, mu, sigma])
    return d_model, q_model


class InfoGAN_Continuous(keras.Model):
    def __init__(self, d_model, g_model, q_model, noise_size, num_classes):
        super(InfoGAN_Continuous, self).__init__()
        self.d_model = d_model
        self.g_model = g_model
        self.q_model = q_model
        self.noise_size = noise_size
        self.num_classes = num_classes

    def compile(self, d_optimizer, g_optimizer, q_optimizer):
        super(InfoGAN_Continuous, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.q_optimizer = q_optimizer

    def create_gen_input(self, batch_size, noise_size=62, code_size=10, n_class=152, seed=None):
        # Create noise input
        noise = tf.random.normal([batch_size, noise_size], seed=seed)

        # Create Code input
        code = tf.random.normal([batch_size, code_size], mean=0, stddev=0.2, seed=seed, dtype=tf.dtypes.float32)

        # Create categorical latent code
        label = tf.random.uniform([batch_size], minval=0, maxval=152, dtype=tf.int32, seed=seed)
        label = tf.one_hot(label, depth=n_class)

        return label, code, noise

    def concat_inputs(self, input):
        concat_input = keras.layers.Concatenate()(input)
        return concat_input

    def train_step(self, real_image_batch):
        # Define loss functions

        binary_loss = keras.losses.BinaryCrossentropy()
        categorical_loss = keras.losses.CategoricalCrossentropy()

        # half batch for trainign discriminator and batch for training generator and auxiliary model
        batch = tf.shape(real_image_batch)[0]

        # Create generator input

        g_label, c_1, g_noise = self.create_gen_input(batch, self.noise_size, self.num_classes, seed=None)
        g_input = self.concat_inputs([g_label, c_1, g_noise])

        with tf.GradientTape() as d_tape:
            self.d_model.trainable = True
            d_tape.watch(self.d_model.trainable_variables)

            # Train discriminator using half batch real images
            y_disc_real = tf.ones((batch, 1))
            d_real_output = self.d_model(real_image_batch, training=True)
            d_loss_real = binary_loss(y_disc_real, d_real_output)

            # Train discriminator using half batch fake images
            y_disc_fake = tf.zeros((batch, 1))

            # Create fake image batch
            print(g_input)
            fake_image_batch = self.g_model(g_input, training=True)
            d_fake_output = self.d_model(fake_image_batch, training=True)
            d_loss_fake = binary_loss(y_disc_fake, d_fake_output)
            d_loss = d_loss_real + d_loss_fake

        # Calculate gradients
        d_gradients = d_tape.gradient(d_loss, self.d_model.trainable_variables)

        # Optimize
        self.d_optimizer.apply_gradients(zip(d_gradients, self.d_model.trainable_variables))

        with tf.GradientTape as g_tape, tf.GradientTape as q_tape:
            # Create generator input
            g_label, c_1, g_noise = self.create_gen_input(batch * 2, self.noise_size, self.num_classes)
            g_input = self.concat_inputs([g_label, c_1, g_noise])
            g_tape.watch(self.g_model.trainable_variables)
            q_tape.watch(self.q_model.trainable_variables)

            # Create fake image batch
            fake_image_batch = self.g_model(g_input, training=True)
            d_fake_output = self.d_model(fake_image_batch, training=True)

            # Generator Image loss
            y_gen_fake = tf.ones((batch * 2, 1))
            g_img_loss = binary_loss(y_gen_fake, d_fake_output)

            # Auxiliary loss
            cat_output, mu, sigma = self.q_model(fake_image_batch, training=True)

            # Categorical loss
            cat_loss = categorical_loss(g_label, cat_output)

            # Use Gaussian distributions to represent the output
            dist = tfp.distributions.Normal(loc=mu, scale=sigma)

            # Losses (negative log probability density function as we want to maximize the proabbility)
            c_1_loss = tf.reduce_mean(-dist.log_prob(c_1))

            # Generator total loss
            g_loss = g_img_loss + (cat_loss + 0.1 * c_1_loss)

            # Auxiliary function loss
            q_loss = (cat_loss + 0.1 * c_1_loss)

        # Calculate gradients
        # We do not want to modify the neurons in the discriminator when training the generator
        self.d_model.trainable = False
        g_gradients = g_tape.gradient(g_loss, self.g_model.trainable_variables)
        q_gradients = q_tape.gradient(q_loss, self.q_model.trainable_variables)

        # Optimize
        self.g_optimizer.apply_gradients(zip(g_gradients, self.g_model.trainable_variables))
        self.q_optimizer.apply_gradients(zip(q_gradients, self.q_model.trainable_variables))

        return {"d_loss_real": d_loss_real, "d_loss_fake": d_loss_fake, "g_img_loss": g_img_loss, "cat_loss": cat_loss,
                "c_1_loss": c_1_loss}


def load_real_image(batch_size=32):
    cur_path = os.getcwd()
    classes = [x[0].split('\\')[-1] for x in os.walk(cur_path)]
    data = []
    labels = []
    for i in classes:
        try:
            images = os.listdir(i)
            for a in images:
                if not a.endswith('svg'):
                    image = Image.open(i + '\\' + a)
                    try:
                        image = image.resize((60, 60))
                        image = np.array(image)
                        image = image.reshape((60, 60, 3))

                        data.append(image)
                        labels.append(classes.index(i))
                    except:
                        pass
        except:
            pass
    data = np.array(data)
    label = np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    # Add the colour channel - change to 4D tensor, and convert the data type to 'float32'
    train_images = x_train.reshape((x_train.shape[0], 60, 60, 3)).astype('float32')
    y_train = to_categorical(y_train, 152)

    train_images = (train_images / 255.0) * 2 - 1
    buffer_size = train_images.shape[0]
    train_images_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)
    return train_images_dataset, y_train


g_model = create_discriminator_continuous()
d_model, q_model = create_discriminator_continuous()

infogan = InfoGAN_Continuous(d_model, g_model, q_model, noise_size=62, num_classes=152)
infogan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=2e-4),
                g_optimizer=keras.optimizers.Adam(learning_rate=5e-4),
                q_optimizer=keras.optimizers.Adam(learning_rate=2e-4))
real_images, y_label = load_real_image(batch_size=32)
infogan.fit(real_images, epochs=80, verbose=1)
