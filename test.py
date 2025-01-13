from sklearn.neighbors import KNeighborsClassifier

import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('Data/frontalface_default.xml')

with open('Data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('Data/face_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Print LABELS content for debugging
print("Original LABELS content:", LABELS[:5])  # Print first 5 for inspection

# Flatten or clean up LABELS
LABELS = [label[0] if isinstance(label, (list, tuple)) else label for label in LABELS]

# Check LABELS structure after cleanup
print("Processed LABELS content:", LABELS[:5])  # Print first 5 again

# Ensure LABELS matches FACES in length
if len(LABELS) > len(FACES):
    LABELS = LABELS[:len(FACES)]
elif len(FACES) > len(LABELS):
    FACES = FACES[:len(LABELS)]

# Ensure labels are consistent
assert len(LABELS) == len(FACES), f"Mismatch: {len(LABELS)} labels and {len(FACES)} faces"

# Preprocess FACES
processed_faces = []
for face in FACES:
    resized_face = cv2.resize(face, (50, 50)).flatten()
    processed_faces.append(resized_face)

FACES = np.array(processed_faces)

# Check consistency
print(f"FACES shape: {FACES.shape}")
print(f"LABELS shape: {np.array(LABELS).shape}")

# Ensure labels are consistent
assert len(LABELS) == len(FACES), "Mismatch between FACES and LABELS"

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

img_background = cv2.imread("background_image.png")

COL_NAMES = ['NAME', 'TIME']

while True:
    ret,frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for x, y, w, h in faces:
        crop_img = gray[y:y+h, x:x+w]  # Use grayscale crop
        try:
            # Resize and reshape the cropped image
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        except Exception as e:
            print(f"Error processing cropped image: {e}")
            continue

        # Check for shape mismatch
        if resized_img.shape[1] != FACES.shape[1]:
            print(f"Shape mismatch: resized {resized_img.shape[1]}, expected {FACES.shape[1]}")
            cv2.putText(frame, "Shape mismatch!", (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            continue

        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        file_exists = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        
        attendance = [str(output[0]), str(timestamp)]

    resized_frame = cv2.resize(frame, (329, 329))
    img_background[162:162+329, 55:55+329] = resized_frame
    
    cv2.imshow("frame", img_background)
    key = cv2.waitKey(1)
    
    if key == ord('a'):
        if file_exists:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(attendance)
            csv_file.close()
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csv_file.close()
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
