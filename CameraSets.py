#%%
import cv2
import numpy as np
import os
import time
import shutil

GESTURES = ["fist", "palm", "peace", "thumbs_up"]
SAVE_INTERVAL = 0.1
IMG_SIZE = 128

os.makedirs("dataset", exist_ok=True)
for g in GESTURES:
    os.makedirs(f"dataset/{g}", exist_ok=True)



def get_silhouette_and_bbox(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))

    # Clean mask
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find hand contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None, None, None

    hand = max(contours, key=cv2.contourArea)

    # Make silhouette (same as dataset & live)
    silhouette = np.zeros_like(mask)
    cv2.drawContours(silhouette, [hand], -1, 255, -1)

    # Bounding box
    x, y, w, h = cv2.boundingRect(hand)

    #Changing to bounding square 
    

    return silhouette, x, y, w, h, mask



def crop_silhouette_exact(silhouette, x, y, w, h):
    if silhouette is None:
        return None

    # Crop the bounding box
    roi = silhouette[y:y+h, x:x+w]

    # Resize to fixed 128x128 (same as live)
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    return roi


#recording loop 

cap = cv2.VideoCapture(0)

print("Live mode ON")
print("Press 'q' to switch to silhouette mode.")

mode = "live"
recording = False
current_gesture = None
last_save = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # first im going to preview the frame

    if mode == "live":
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            mode = "silhouette"
            cv2.destroyWindow("Camera")
            print("\nSilhouette mode ON")
            print("Press 1/2/3/4 to start/stop datasets.")
            print("Press 'q' again to quit silhouette mode.")
        continue

 
    # masked up mode and recording has now started 
  
    silhouette, x, y, w, h, mask = get_silhouette_and_bbox(frame)

    # If no hand found
    if silhouette is None:
        preview = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.imshow("Silhouette Preview", preview)
    else:
        cropped = crop_silhouette_exact(silhouette, x, y, w, h)
        cv2.imshow("Silhouette Preview", cropped)

    key = cv2.waitKey(1) & 0xFF

    # Quit silhouette mode
    if key == ord('q'):
        print("Exiting silhouette mode…")
        break

    # Start/stop gesture recording
    for i, g in enumerate(GESTURES):
        if key == ord(str(i+1)):
            recording = not recording
            current_gesture = g if recording else None

            if recording:
                print(f"STARTED recording: {g}")
            else:
                print(f" STOPPED recording: {g}")

    # Save frames (ONLY cropped version)
    if recording and silhouette is not None:
        if time.time() - last_save >= SAVE_INTERVAL:
            save_img = crop_silhouette_exact(silhouette, x, y, w, h)
            filename = f"dataset/{current_gesture}/{int(time.time()*1000)}.png"
            cv2.imwrite(filename, save_img)
            last_save = time.time()
            print(f"[Saved] {filename}")

    # Delete gesture folder (press d then gesture #)
    if key == ord('d'):
        next_key = cv2.waitKey(0) & 0xFF
        index = next_key - ord('1')
        if 0 <= index < len(GESTURES):
            gesture_to_delete = GESTURES[index]
            folder = f"dataset/{gesture_to_delete}"
            shutil.rmtree(folder)
            os.makedirs(folder)
            print(f" Deleted all images from folder: {folder}")

cap.release()
cv2.destroyAllWindows()

# %%

root = "dataset"

print("\nImage counts in dataset/:")
for gesture in GESTURES:
    folder = os.path.join(root, gesture)
    if not os.path.isdir(folder):
        print(f"{gesture}: (folder missing)")
        continue

    files = [
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) 
           and f.lower().endswith(".png")
    ]
    print(f"{gesture}: {len(files)} images")






# %%
import cv2
import numpy as np
import os
import shutil

SOURCE = "dataset"                 
DEST = "dataset_clean"             
IMG_SIZE = 128


if os.path.exists(DEST):
    shutil.rmtree(DEST)
os.makedirs(DEST)

def crop_silhouette(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
   
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours (hand is largest white region)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return None
    
    hand = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(hand)

    # Crop
    roi = img[y:y+h, x:x+w]

    # Resize 
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))

    return roi

print("Processing dataset")

# Walk through each gesture folder
for gesture in sorted(os.listdir(SOURCE)):
    gesture_folder = os.path.join(SOURCE, gesture)
    if not os.path.isdir(gesture_folder):
        continue

    out_folder = os.path.join(DEST, gesture)
    os.makedirs(out_folder, exist_ok=True)

    for fname in os.listdir(gesture_folder):
        fpath = os.path.join(gesture_folder, fname)

        processed = crop_silhouette(fpath)
        if processed is None:
            print(f"Skipping bad image: {fpath}")
            continue

        out_path = os.path.join(out_folder, fname)
        cv2.imwrite(out_path, processed)

print(f"Cleaned dataset saved to: {DEST}")

#%%
root = "dataset_clean"
for gesture in os.listdir(root):
    path = os.path.join(root, gesture)
    count = len(os.listdir(path))
    print(f"{gesture}: {count} images")




# %%
import cv2
import numpy as np

IMG_SIZE = 128 



def get_silhouette(frame):
    ycrcb=cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))

    mask= cv2.GaussianBlur(mask, (5,5), 0)
    mask = cv2.erode(mask, None, iterations=1)
    mask=   cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None, None, mask

    hand = max(contours, key=cv2.contourArea)

    # Create silhouette 
    silhouette = np.zeros_like(mask)
    cv2.drawContours(silhouette, [hand], -1, 255, -1)

    # Bounding box from og contour
    x, y, w, h = cv2.boundingRect(hand)

    return silhouette, x, y, w, h, mask




# Crop masked 128x128

def crop_silhouette(silhouette, x, y, w, h):
    if silhouette is None:
        return None

    # Crop ROI 
    roi = silhouette[y:y+h, x:x+w]

    # Resize 
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))

    return roi



# Loop for live camera 
cap = cv2.VideoCapture(0)

print("Press q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break




   
    silhouette, x, y, w, h, mask = get_silhouette(frame)
    cropped = None
    if silhouette is not None:
        cropped = crop_silhouette(silhouette, x, y, w, h)

    display = frame.copy()
    if silhouette is not None:
        cv2.rectangle(display, (x, y), (x+w, y+h), (0,255,0), 2)





    cv2.imshow("Live Camera ", display)
    cv2.imshow("Silhouette ", silhouette if silhouette is not None else mask)
    if cropped is not None:
        cv2.imshow("Cropped ", cropped)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# %%
