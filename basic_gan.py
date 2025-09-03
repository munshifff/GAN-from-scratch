import numpy as np
import matplotlib.pyplot as py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_generator(latent_dim, output_dim=1):
    model = Sequential()
    model.add(Dense(10, input_dim=latent_dim, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))  # output_dim should match discriminator input_dim
    return model

def build_discriminator(input_dim=1):
    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def generate_real_samples(n):
    x = np.random.randn(n)
    y = np.ones((n, 1))
    return x, y

def generate_latent_points(latent_dim, n):
    return np.random.randn(n, latent_dim)

def train_gan(gan, generator, discriminator, latent_dim, n_epochs=100, n_batch=128):
    half_batch = n_batch // 2

    for epoch in range(n_epochs):

        x_real, y_real = generate_real_samples(half_batch)
        d_loss_real, _ = discriminator.train_on_batch(x_real, y_real)
        
        x_fake = generator.predict(generate_latent_points(latent_dim, half_batch))
        y_fake = np.zeros((half_batch, 1))
        d_loss_fake, _ = discriminator.train_on_batch(x_fake, y_fake)
        
        x_gan = generate_latent_points(latent_dim, n_batch)
        y_gan = np.ones((n_batch, 1))
        g_loss = gan.train_on_batch(x_gan, y_gan)
        
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch {epoch+1}, D Loss Real: {d_loss_real}, D Loss Fake: {d_loss_fake}, G Loss: {g_loss}')


latent_dim = 5

discriminator = build_discriminator()

generator = build_generator(latent_dim)

gan_model = build_gan(generator, discriminator)

train_gan(gan_model, generator, discriminator, latent_dim)