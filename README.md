# Simple GAN

A minimal Generative Adversarial Network (GAN) implemented in Keras.

## Architecture

![GAN Architecture](https://miro.medium.com/v2/resize:fit:800/1*W9fwrLqgW4fItWj4IuE3tw.png)

- **Generator (G):** takes random noise and creates fake samples  
- **Discriminator (D):** takes real and fake samples, tries to classify them  
- Both compete in a minimax game until Generator produces realistic data  
