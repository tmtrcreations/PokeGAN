#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 2018

Author: 
Modified By: tmtrcreations
"""

# --------------------------
# Import the needed modules
# --------------------------

from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization, Activation, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# -----------
#  GAN Class 
# -----------

class GAN():
    
    # --------------------
    #  Define Initializer 
    # --------------------
    
    def __init__(self):
        self.img_rows = 40
        self.img_cols = 40
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer_d = SGD(0.001)
        optimizer_g = Adam(0.002)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer_d,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        validity = self.discriminator
        validity.trainable = False
        # The discriminator takes generated images as input and determines validity
        validity = validity(img)
        

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer_g)

    # ------------------
    #  Define Generator 
    # ------------------

    def build_generator(self):

        model = Sequential()
        
        model.add(Dense(128 * 5 * 5, activation='relu', input_dim=self.latent_dim))
        model.add(Reshape((5, 5, 128)))
        model.add(UpSampling2D())
        
        model.add(Conv2D(64, kernel_size=2, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.9))  
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        
        model.add(Conv2D(32, kernel_size=2, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        
        model.add(Conv2D(self.channels, kernel_size=2, strides=1, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)
    
    # ----------------------
    #  Define Discriminator 
    # ----------------------

    def build_discriminator(self):

        model = Sequential()
        
        model.add(Conv2D(32, (2, 2), padding='valid', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.9))
        model.add(AveragePooling2D((2, 2)))
        
        model.add(Conv2D(64, (2, 2), padding='valid'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.9))
        model.add(AveragePooling2D((2, 2)))
        
        model.add(Conv2D(128, (2, 2), padding='valid'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.9))
        model.add(AveragePooling2D((2, 2)))
        
        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)
    
    # -----------------
    #  Define Training 
    # -----------------

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train = 255*np.ones((1722*2, self.img_rows, self.img_cols, self.channels))
        for img_ind in range(1, 1723):
            img = Image.open('../../Images/Training_Images/Single_Sprites/img' + str(img_ind) + '.png')
            X_train[img_ind-1, 5:35, :, :] = np.array(img)
        for img_ind in range(1, 1723):
            img = Image.open('../../Images/Training_Images/Single_Sprites_Flipped/img_flipped' + str(img_ind) + '.png')
            X_train[1722+img_ind-1, 5:35, :, :] = np.array(img)

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            imgs = imgs + (0.5*(np.cos(np.pi*epoch/10000)+1)/2)*np.random.normal(0, 1, imgs.shape)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    # --------------------------
    #  Define Sample Generation 
    # --------------------------

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :, :, :])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("../../Images/Generated_Sprites/%d.png" % epoch)
        plt.close()

# ------
#  Main
# ------
        
if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=100000, batch_size=64, sample_interval=200)
