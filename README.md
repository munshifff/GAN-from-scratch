# Simple GAN

A minimal Generative Adversarial Network (GAN) implemented in Keras.

## Architecture

![GAN Architecture](https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/master/gan_faces/assets/gan.png)


- **Generator (G):** takes random noise and creates fake samples  
- **Discriminator (D):** takes real and fake samples, tries to classify them  
- Both compete in a minimax game until Generator produces realistic data  
