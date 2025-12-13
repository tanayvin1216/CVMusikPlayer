# CVMusikPlayer: A real time computer vision music player

# ⭐️ Overview

> *This project explores real-time hand gesture recognition using Computer Vision and Machine Learning. A Convolutional Neural Netowrk (CNN) is trained on a custom dataset to classify hand gestures from image data, enabling gesture based interaction with a small scale music player. 

Through computer vision and YCbCr color processing, image footage from a 2MP global shutter camera was converted into black and white footage. Using these images a dataset of approximately 2000 images was constructed. Using this custom dataset, a CNN was trained and evaluated. Live testing showed that the CNN was able to correctly identify hand gestures and play music according to the gesture. The CNN was later integrated with PyGame Mixer for efficient music playback.

This system includes a full pipleine for custom dataset preperation, model training and evalutation. One of the primary components is also ensuring clean data organization and reproducibility. This inital implementation focusses on a small set of gestures but this project can easily be scaled to additonal gestures and real-time applications.*


## 🌟 Goals

One of my dad's favorite song, "Yeah" by Usher, starts off with the lyrics "Peace up A town". Everytime we play it on a drive, my dad always makes sure to hit the Peace sign. 

The goal of this project was to create a Computer Vision (CV) system capapble of recognizing simple hand gestures starting with the Peace sign. By translating a hand gesture into a cue to play music, this project explores gesture recognition as an intuitive and expressive human–computer interaction. It also served as something I wanted to impress my dad with for his birthday. 


### ✍️ Project structure

- CameraSets.py        # Dataset loading and preprocessing functionalities 
- model.py             # CNN model definition and training logic
- dataset/             # Raw gesture data (optional / original source)
- dataset_clean/       # Cleaned and labeled gesture images
- sounds/              # audio files 




## ℹ️ Dataset and Model 
### Dataset
The dataset consists of labeled images of hand gestures organized by class. Images were captured every .1 seconds during recording allowing for quick and seamless dataset creation. The dataset includes images of 128x128 input size.

The cleaned dataset is used to ensure consistent input dimensions and improved model performance. This separation also allows reproducibility and future dataset augmentation.

Each captured frame undergoes the following steps:
1. **Frame Capture**
   Raw frames are captured from a live camera stream while the user performs a
   specific gesture. Frames are saved at regular intervals to build a diverse set
   of examples for each gesture class.

2. **Region of Interest Extraction**
   To reduce background interference, frames are cropped to focus on the hand
   region. This step limits irrelevant visual information and improves feature
   learning efficiency.

3. **Image Resizing**
   All images are resized to a fixed resolution to ensure consistent input
   dimensions across the dataset, a requirement for CNN-based models.

4. **Normalization**
   Pixel values are normalized to a common scale, stabilizing training and
   improving convergence during optimization.

5. **Data Cleaning**
   Blurry frames, duplicate images, and incorrectly labeled samples are removed.
   This results in a cleaned dataset (`dataset_clean/`) that is used exclusively
   for model training and evaluation.

This preprocessing pipeline converts raw camera frames into standardized tensors
that can be reliably ingested by the convolutional neural network.


<img width="350" height="395" alt="Screenshot 2025-12-13 at 11 27 17 AM" src="https://github.com/user-attachments/assets/ad34301e-8a34-41ff-a325-3712d901a8e0" />

<img width="350" height="395" alt="PalmVsFist_CVMusikPlayer" src="https://github.com/user-attachments/assets/93ef0e0e-f820-4d10-81db-2a6622b150a3" />


### Model
The gesture recognition model is implemented using a convolutional neural network(CNN).The architecture is designed to extract spatial features from input images and classify them into predefined gesture categories.

Key characteristics:
- Image-based CNN architecture 
- Supervised training on labeled gesture data
- Designed for real time classification and useage
- inegrated with PyGame Mixer allowing for efficent muisc playback


## Images of Model prediction during Live Demo
<img width="200" height="400" alt="Screenshot 2025-12-03 at 4 55 41 PM" src="https://github.com/user-attachments/assets/73c996a2-005d-4328-8ccf-8bba9b1a848c" />

<img width="200" height="427" alt="Screenshot 2025-12-13 at 11 42 59 AM" src="https://github.com/user-attachments/assets/56cc4365-4cd0-48b1-80fe-8efa4e54575b" />

<img width="200" height="414" alt="Screenshot 2025-12-13 at 11 45 21 AM" src="https://github.com/user-attachments/assets/ee1d2440-3a64-47db-b2de-c8022521bc38" />

<img width="200" height="400" alt="Screenshot 2025-12-13 at 11 47 32 AM" src="https://github.com/user-attachments/assets/b4df0c3f-166e-4ae3-b134-3060c5d33520" />




### Architecture
The network consists of three convolutional layers followed by fully connected layers for classification. Each convolutional layer applies a 3×3 filter, a ReLU
activation, and max pooling to reduce spatial dimensions while retaining important features.

- **Conv Layer 1:**  
  1 → 16 filters → ReLU → MaxPool  
  Output shape: **(16, 64, 64)**

- **Conv Layer 2:**  
  16 → 32 filters → ReLU → MaxPool  
  Output shape: **(32, 32, 32)**

- **Conv Layer 3:**  
  32 → 64 filters → ReLU → MaxPool  
  Output shape: **(64, 16, 16)**

After feature extraction, the output is flattened and passed through fully connected layers to produce gesture class predictions.

### Training
ReLU activations are used throughout the network, and the model is trained using cross-entropy loss, which is well-suited for gesture classification tasks.
This architecture balances accuracy and efficiency and can be extended to support additional gestures or real-time inference.



## 🚀 Future Work

Future development will focus on expanding both the technical capabilities and practical applications of the system. Planned improvements include support for a broader set of hand gestures and mapping those gestures to meaningful actions rather than simple demonstrations.

One direction is integration with music-streaming APIs and voice-powered agents, allowing gesture recognition to function as part of a more seamless music control system. This would enable users to interact naturally with media without relying on physical interfaces.

Although this project began as an exploratory and creative experiment, gesture recognition has significant real-world potential. Lightweight hand-gesture models can improve accessibility by enabling alternative input methods for individuals with limited mobility, reduce friction in human–computer interaction, and support more intuitive control in everyday environments. 
















