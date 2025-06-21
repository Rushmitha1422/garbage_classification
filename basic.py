import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Set path to dataset
DATASET_DIR = "C:/Users/USER/OneDrive/Desktop/edunet_internship/TrashType_Image_Dataset"

IMAGE_SIZE = (64, 64)

# 2. Load images and extract features
def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, IMAGE_SIZE)
    # Convert to HSV and compute histogram (color-based features)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [4, 4, 4], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

features = []
labels = []

classes = os.listdir(DATASET_DIR)
for label in classes:
    class_dir = os.path.join(DATASET_DIR, label)
    for file in os.listdir(class_dir):
        path = os.path.join(class_dir, file)
        feature = extract_features(path)
        features.append(feature)
        labels.append(label)

print(f"Extracted features from {len(features)} images.")

# 3. Split data
X = np.array(features)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train a basic ML model (Random Forest)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()