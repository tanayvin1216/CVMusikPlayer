#CV Musik Player: A real time computer vision model


---Overview---
This project explores real-time hand gesture recognition using Computer Vision and Machine Learning. A Convolutional Neural Netowrk (CNN) is trained on a custom dataset to classify hand gestures from image data, enabling gesture based interaction with a small scale music player. 

Through computer vision and YCbCr color processing, image footage from a 2MP global shutter camera was converted into black and white footage. Using these images a dataset of approximately 2000 images was constructed. Using this custom dataset, a CNN was trained and evaluated. Live testing showed that the CNN was able to correctly identify hand gestures and play music according to the gesture. The CNN was later integrated with PyGame Mixer for efficient music playback.

This system includes a full pipleine for custom dataset preperation, model training and evalutation. One of the primary components is also ensuring clean data organization and reproducibility. This inital implementation focusses on a small set of gestures but this project can easily be scaled to additonal gestures and real-time applications.  



---Goal---
One of my dad's favorite song, "Yeah" by Usher, starts off with the lyrics "Peace up A town". Everytime we play it on a drive, my dad always makes sure to hit the Peace sign. 

The goal of this project was to create a Computer Vision (CV) system capapble of recognizing simple hand gestures starting with the Peace sign. By translating a hand gesture into a cue to play music, this project explores gesture recognition as an intuitive and expressive human–computer interaction. It also served as something I wanted to impress my dad with for his birthday. 



---Project Structure---
├─ CameraSets.py        # Dataset loading and preprocessing functionalities 
├─ model.py             # CNN model definition and training logic
├─ dataset/             # Raw gesture data (optional / original source)
├─ dataset_clean/       # Cleaned and labeled gesture images
├─ sounds/              # audio files 

---Dataset---
The dataset consists of labeled images of hand gestures organized by class. Images were captured every .1 seconds during recording allowing for quick and seamless dataset creation. The dataset includes images of 128x128 input size.

The cleaned dataset is used to ensure consistent input dimensions and improved model performance. This separation also allows reproducibility and future dataset augmentation.

---Model---
The gesture recognition model is implemented using a convolutional neural network(CNN).The architecture is designed to extract spatial features from input images and classify them into predefined gesture categories.

Key characteristics:
- Image-based CNN architecture 
- Supervised training on labeled gesture data
- Designed for real time classification and useage
- inegrated with PyGame Mixer allowing for efficent muisc playback


---Future Work---
Future development will focus on expanding both the technical capabilities and practical applications of the system. Planned improvements include support for a broader set of hand gestures and mapping those gestures to meaningful actions rather than simple demonstrations.

One direction is integration with music-streaming APIs and voice-powered agents, allowing gesture recognition to function as part of a more seamless music control system. This would enable users to interact naturally with media without relying on physical interfaces.

Although this project began as an exploratory and creative experiment, gesture recognition has significant real-world potential. Lightweight hand-gesture models can improve accessibility by enabling alternative input methods for individuals with limited mobility, reduce friction in human–computer interaction, and support more intuitive control in everyday environments. 