#%%
#Imports 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_curve
import pandas as pd
# %%
import os
import shutil
import random

SOURCE = "dataset"         
TRAIN = "dataset_train"    
TEST = "dataset_test"      

TRAIN_SPLIT = 0.8           # 70% train / 20% test


def create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)



create_dir(TRAIN)
create_dir(TEST)

# 2. Get gesture folders
gestures = sorted(os.listdir(SOURCE))

for gesture in gestures:
    gesture_path = os.path.join(SOURCE, gesture)
    if not os.path.isdir(gesture_path):
        continue

    # Create gesture folder in train and test
    train_gesture_path = os.path.join(TRAIN, gesture)
    test_gesture_path = os.path.join(TEST, gesture)
    os.makedirs(train_gesture_path)
    os.makedirs(test_gesture_path)

    # List all images in gesture folder
    files = sorted(os.listdir(gesture_path))
    random.shuffle(files)

    # Determine split index
    split_idx = int(len(files) * TRAIN_SPLIT)

    train_files = files[:split_idx]
    test_files = files[split_idx:]

    # Copy files into train/test folders
    for f in train_files:
        src = os.path.join(gesture_path, f)
        dst = os.path.join(train_gesture_path, f)
        shutil.copy(src, dst)

    for f in test_files:
        src = os.path.join(gesture_path, f)
        dst = os.path.join(test_gesture_path, f)
        shutil.copy(src, dst)

    print(f"{gesture}: {len(train_files)} train, {len(test_files)} test")

print("\n Done! Train/test dataset created successfully.")
print(f" Train folder: {TRAIN}")
print(f" Test folder:  {TEST}")

#%%

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

IMG_SIZE = 128
GESTURES = ["fist", "palm", "peace", "thumbs_up"]   # all four classes 


# Load dataset 

def load_tensor_dataset(path):
    X = []
    y = []
    for label, gesture in enumerate(GESTURES):
        folder = os.path.join(path, gesture)
        for file in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = img.astype("float32") / 255.0
            X.append(img)
            y.append(label)

    X = np.array(X).reshape(-1, 1, IMG_SIZE, IMG_SIZE)
    y = np.array(y)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# LOAD TRAIN / TEST DATA FROM FOLDERS
X_train, y_train = load_tensor_dataset("dataset_train")
X_test,  y_test  = load_tensor_dataset("dataset_test")

print("Train size:", X_train.shape)
print("Test size:",  X_test.shape)





class GestureCNN(nn.Module):
    def __init__(self):
        super(GestureCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        
        self.dropout = nn.Dropout(p=0.3)   
        
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 4)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = self.pool(F.relu(self.conv3(x)))  

        x = x.view(-1, 64 * 16 * 16)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = GestureCNN()
print(model)



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 32
epochs = 15


#Training Loop

for epoch in range(epochs):
    permutation = torch.randperm(X_train.size(0))

    for i in range(0, X_train.size(0), batch_size):
        idx = permutation[i:i+batch_size]
        inputs = X_train[idx]
        labels = y_train[idx]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")


#Eval

model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, preds = torch.max(test_outputs, 1)
    acc = (preds == y_test).float().mean().item()

print(f"\nTest Accuracy: {acc*100:.2f}%")



torch.save(model.state_dict(), "gesture_cnn_4class.pth")
print("Model saved → gesture_cnn_4class.pth")

#%% LIVE CAMERA TEST FOR TS CNN
import pygame 
import cv2
import numpy as np
import torch
import torch.nn.functional as F

pygame.mixer.init()


IMG_SIZE = 128
GESTURES = ["fist", "palm", "peace","thumbs_up"]   
model = GestureCNN()
model.load_state_dict(torch.load("gesture_cnn_4class.pth"))
model.eval()  

channel = pygame.mixer.Channel(0)


sound_map = {
    "thumbs_up": pygame.mixer.Sound("sounds/asap.wav"),
    "palm":      pygame.mixer.Sound("sounds/holdup.wav"),
    "fist":      pygame.mixer.Sound("sounds/power.wav"),
    "peace":     pygame.mixer.Sound("sounds/yeah.wav")
}

import time

last_gesture_played = None
last_play_time = 0
COOLDOWN = 600.0  #stopping sound from blowing computer bruh
audio_enabled = True

def play_gesture_sound(gesture):
    global last_gesture_played, last_play_time

    if not audio_enabled:
        return  

    if gesture not in sound_map:
        return

    now = time.time()

  
    if gesture == last_gesture_played and channel.get_busy():
        return


    if gesture == last_gesture_played and (now - last_play_time) < COOLDOWN:
        return


    channel.stop()  
    channel.play(sound_map[gesture])
    last_gesture_played = gesture
    last_play_time = now

    print(f"Now Playing: {gesture}")

# Keep track of last played gesture (prevents repeating)
last_gesture = None

print("LOADING SOUNDS…")
print("CWD:", os.getcwd()) 

for name, sound in sound_map.items():
    print("Loaded:", name, "→", sound)

try:
    pygame.mixer.init()
    print("mixer ready boss")
except Exception as e:
    print("MIXER is cooked:", e)



def get_silhouette_and_bbox(frame):

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    

    mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))

    # Clean up mask
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    # Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None, None, mask
    
    hand = max(contours, key=cv2.contourArea)


    silhouette = np.zeros_like(mask)
    cv2.drawContours(silhouette, [hand], -1, 255, -1)

    # Bounding box
    x, y, w, h = cv2.boundingRect(hand)
    return silhouette, x, y, w, h, mask


#changing image format for the model 
def crop_for_model(silhouette, x, y, w, h):
    if silhouette is None:
        return None
    roi = silhouette[y:y+h, x:x+w]


    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return roi


#live video loo

cap = cv2.VideoCapture(0)
print("Live gesture testing enabled — press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    silhouette, x, y, w, h, mask = get_silhouette_and_bbox(frame)
    display= frame.copy()

    if silhouette is not None:
        cropped = crop_for_model(silhouette, x, y, w, h)
        
        if cropped is not None:
    
            inp= cropped.astype("float32") / 255.0
            inp=torch.tensor(inp).unsqueeze(0).unsqueeze(0)  


            with torch.no_grad():
                logits=model(inp)
                probs=F.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, 1)
            
            gesture=GESTURES[pred_idx.item()]
            confidence=float(conf.item() * 100)
            
            if confidence > 50:   
                play_gesture_sound(gesture)

           

            # Draw bounding box + label
            cv2.rectangle(display, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(display, f"{gesture} ({confidence:.1f}%)",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,0), 2)


            cv2.imshow("Model Input (128x128)", cropped)


    if silhouette is not None:
        cv2.imshow("Silhouette", silhouette)
    else:
        cv2.imshow("Silhouette", mask)


    cv2.imshow("Live Gesture Prediction", display)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# %%
import matplotlib.pyplot as plt

for label, gesture in enumerate(GESTURES):
    print("Class:", gesture)
    idxs = (y == label).nonzero().squeeze()[:5]
    for i in idxs:
        plt.imshow(x[i][0], cmap='gray')
        plt.show()
# %%
from sklearn.model_selection import KFold

X, y = load_tensor_dataset("dataset_train")
print("Total samples:", X.shape[0])


kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_accuracies = []

fold_num = 1
for train_idx, val_idx in kf.split(X, y):
    print(f"Fold {fold_num}/5")

    X_train_cv = X[train_idx]
    y_train_cv = y[train_idx]
    X_val_cv = X[val_idx]
    y_val_cv = y[val_idx]


    fold_model = GestureCNN()
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(fold_model.parameters(), lr=0.0001)

    epochs= 10
    batch_size = 32

    for epoch in range(epochs):
        perm = torch.randperm(X_train_cv.size(0))

        for i in range(0, X_train_cv.size(0), batch_size):
            idx = perm[i:i+batch_size]
            inputs  = X_train_cv[idx]
            labels  = y_train_cv[idx]

            optimizer.zero_grad()
            outputs = fold_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"  Fold {fold_num} Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")


    fold_model.eval()
    with torch.no_grad():
        out   = fold_model(X_val_cv)
        _, pr = torch.max(out, 1)
        acc   = (pr == y_val_cv).float().mean().item() * 100

    print(f"✔ Fold {fold_num} Accuracy: {acc:.2f}%")
    cv_accuracies.append(acc)
    fold_num += 1

print("CROSS-VAL SUMMARY")
print("accuracies:", cv_accuracies)
print(f"Mean: {np.mean(cv_accuracies):.2f}%")
print(f"Std:  {np.std(cv_accuracies):.2f}%")

# %%
