from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU, Activation, Concatenate, BatchNormalization, Dropout, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from settings import MainPathData
import matplotlib.pyplot as plt
from time import time
import numpy as np
import datetime


class UrbanPlanningGAN(object):

    def __init__(self):

        self.img_rows = 256
        self.img_cols = 384
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.x_train: np.array = np.array([])
        self.y_train: np.array = np.array([])

        self.gf = 64
        self.df = 16

        self.optimizer = Adam(0.0002, 0.5)

        self.d_model = self.define_discriminator(self.img_shape)
        self.g_model = self.define_generator(self.img_shape)
        self.gan_model = self.define_gan(self.g_model, self.d_model, self.img_shape)

        self.dloss = []
        self.dacc = []
        self.gloss = []

    def load_data(self):

        cur_time = time()
        print('Loading x_train...')
        self.x_train = np.load(MainPathData.data.arrays.joinpath('x_train.npy'))
        print('Loading y_train...')
        self.y_train = np.load(MainPathData.data.arrays.joinpath('y_train.npy'))
        print(f'Dataset loaded in {round(time() - cur_time, 2)}sec.')

        return self

    def define_generator(self, image_shape):

        def conv2d(layer_input, filters, f_size=(1, 1)):

            d = Conv2D(filters, kernel_size=f_size, padding='same')(layer_input)
            d = BatchNormalization(momentum=0.8)(d)
            out = LeakyReLU(alpha=0.2)(d)
            d = Conv2D(filters, kernel_size=f_size, padding='same')(out)
            d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            d = Conv2D(filters, kernel_size=f_size, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([out, d])
            d = LeakyReLU(alpha=0.2)(d)
            d = MaxPooling2D((2, 2))(d)

            return d

        def deconv2d(layer_input, skip_input, filters, f_size=(1, 1)):

            u = UpSampling2D(size=(2, 2))(layer_input)
            u = Conv2D(filters, kernel_size=f_size, padding='same')(u)
            u = BatchNormalization(momentum=0.8)(u)
            out = Activation('relu')(u)
            u = Conv2D(filters, kernel_size=f_size, padding='same')(out)
            u = BatchNormalization(momentum=0.8)(u)
            u = Activation('relu')(u)
            u = Conv2D(filters, kernel_size=f_size, padding='same')(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Add()([out, u])
            u = Activation('relu')(u)
            u = Concatenate()([u, skip_input])

            return u

        # Image input
        d0 = Input(shape=image_shape)
        # d0 = Conv2D(64, (3,3), padding='same')(d0)
        # d0 = LeakyReLU(alpha=0.2)(d0)

        # Downsampling
        d1 = conv2d(d0, self.gf, f_size=(3, 3))  # (128, 192, 64)
        d2 = conv2d(d1, self.gf * 2, f_size=(5, 5))  # (64, 96, 128)
        d3 = conv2d(d2, self.gf * 4, f_size=(3, 3))  # (32, 48, 256)
        d4 = conv2d(d3, self.gf * 8, f_size=(3, 3))  # (16, 24, 512)
        d5 = conv2d(d4, self.gf * 8, f_size=(3, 3))  # (8, 12, 512)
        d6 = conv2d(d5, self.gf * 8, f_size=(3, 3))  # (4, 6, 512)
        d7 = conv2d(d6, self.gf * 8, f_size=(3, 3))  # (2, 3, 512)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf * 8, f_size=(3, 3))  # (4, 6, 512)
        u2 = deconv2d(u1, d5, self.gf * 8, f_size=(3, 3))  # (8, 12, 512)
        u3 = deconv2d(u2, d4, self.gf * 8, f_size=(3, 3))  # (16, 24, 512)
        u4 = deconv2d(u3, d3, self.gf * 4, f_size=(3, 3))  # (32, 48, 256)
        u5 = deconv2d(u4, d2, self.gf * 2, f_size=(3, 3))  # (64, 96, 128)
        u6 = deconv2d(u5, d1, self.gf, f_size=(3, 3))  # (128, 192, 64)

        u7 = UpSampling2D(size=(2, 2))(u6)  # (256, 384, 512)
        output_img = Conv2D(3, kernel_size=(3, 3), strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def define_discriminator(self, img_shape):

        def d_layer(layer_input, filters, f_size, bn=True):

            d = Conv2D(filters, kernel_size=f_size, padding='same')(layer_input)
            out = LeakyReLU(alpha=0.2)(d)
            if bn:
                out = BatchNormalization(momentum=0.8)(out)
            d = Conv2D(filters, kernel_size=f_size, padding='same')(out)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=f_size, padding='same')(out)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = Add()([out, d])
            d = MaxPooling2D((2, 2))(d)

            return d

        img_a = Input(shape=img_shape)
        img_b = Input(shape=img_shape)

        combined_imgs = Concatenate(axis=-1)([img_a, img_b])

        d1 = d_layer(combined_imgs, self.df, bn=False, f_size=(5, 5))  # (128, 192, 16)
        d2 = d_layer(d1, self.df * 2, f_size=(5, 5))  # (64, 96, 32)
        d3 = d_layer(d2, self.df * 4, f_size=(3, 3))  # (32, 48, 64)
        d4 = d_layer(d3, self.df * 4, f_size=4)

        validity = Conv2D(1, kernel_size=(3, 3), strides=1, padding='same', activation='sigmoid')(d4)
        model = Model([img_a, img_b], validity)
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['accuracy'])

        return model

    def generate_real_samples(self, n_samples, patch_shape):

        ix = np.random.randint(0, self.x_train.shape[0], n_samples)
        x1, x2 = self.x_train[ix], self.y_train[ix]
        y = np.ones((n_samples, patch_shape, 24, 1))

        return [x1, x2], y

    def generate_fake_samples(self, g_model, samples, patch_shape):

        xx = g_model.predict(samples)
        y = np.zeros((len(xx), patch_shape, 24, 1))

        return xx, y

    def define_gan(self, g_model, d_model, image_shape):

        for layer in d_model.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False

        in_src = Input(shape=image_shape)
        gen_out = g_model(in_src)
        dis_out = d_model([in_src, gen_out])  # [real, fake]
        model = Model(in_src, [dis_out, gen_out])  # [image, [discriminator, generator]]

        model.compile(loss=['mse', 'mae'], optimizer=self.optimizer, loss_weights=[1, 100])

        return model

    def summarize_performance(self, g_model):

        image = load_img(MainPathData.base.joinpath('my_city.bmp'), target_size=(256, 384, 3))
        image = img_to_array(image).astype('float32')
        image = (image - 127.5) / 127.5
        image = np.expand_dims(image, axis=0)

        x_fake_b, _ = self.generate_fake_samples(g_model, image, 1)

        image = (image + 1) / 2.0
        x_fake_b = (x_fake_b + 1) / 2.0
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(image[0])
        ax[0].axis('off')
        ax[1].imshow(x_fake_b[0])
        ax[1].axis('off')

        # filename1 = 'plot_%06d.png' % (step + 1)
        # plt.savefig(###CUSTOM PATH###)
        plt.show()
        plt.close()

    def predict(self, img_name):
        image = load_img(img_name, target_size=self.img_shape)
        image = img_to_array(image).astype('float32')
        image = (image - 127.5) / 127.5
        image = np.expand_dims(image, axis=0)
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(image[0])
        ax[0].set_title('Маска')
        ax[1].imshow(np.array(self.gan_model.predict(image)[1][0]))
        ax[1].set_title('Предсказание сети')

    def train(self, n_epochs=80, n_batch=1):

        start_time = datetime.datetime.now()
        n_patch = self.d_model.output_shape[1]
        bat_per_epo = int(len(self.x_train) / n_batch)
        n_steps = bat_per_epo * n_epochs
        for i in range(n_steps):
            [x_real_a, x_real_b], y_real = self.generate_real_samples(n_batch, n_patch)
            x_fake_b, y_fake = self.generate_fake_samples(self.g_model, x_real_a, n_patch)

            d_loss_real = self.d_model.train_on_batch([x_real_a, x_real_b], y_real)
            d_loss_fake = self.d_model.train_on_batch([x_real_a, x_fake_b], y_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            g_loss, _, _ = self.gan_model.train_on_batch(x_real_a, [y_real, x_real_b])
            self.dloss.append(d_loss[0])
            self.dacc.append(100 * d_loss[1])
            self.gloss.append(g_loss)
            elapsed_time = datetime.datetime.now() - start_time
            print("[Step %d / %d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (i, n_steps,
                                                                                     d_loss[0], 100 * d_loss[1],
                                                                                     g_loss,
                                                                                     elapsed_time))
            if i % 5000 == 0:
                filename2 = 'gan_model_%06d.h5' % i
                self.gan_model.save_weights(f'/content/drive/MyDrive/UAI/Thesis/gan_model/{filename2}')
                print('>Saved gan_model: %s' % filename2)
            if (i + 1) % 100 == 0 or i == 0:
                self.summarize_performance(self.g_model)


UPGAN = UrbanPlanningGAN()
UPGAN.load_data()
UPGAN.train(n_epochs=100, n_batch=8)
UPGAN.d_model.save_weights(MainPathData.base.joinpath('d_model.h5'))
UPGAN.g_model.save_weights(MainPathData.base.joinpath('g_model.h5'))
UPGAN.gan_model.save_weights(MainPathData.base.joinpath('gan_model.h5'))
