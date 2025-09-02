# Simple GAN

A minimal Generative Adversarial Network (GAN) implemented in Keras.

## Architecture

![GAN Architecture](https://www.ijraset.com/images/text_version_uploads/imag%201_53059.png)


- **Generator (G):** takes random noise and creates fake samples  
- **Discriminator (D):** takes real and fake samples, tries to classify them  
- Both compete in a minimax game until Generator produces realistic data  
