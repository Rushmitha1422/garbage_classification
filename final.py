import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Set path to dataset
DATASET_DIR = r"C:\Users\USER\OneDrive\Desktop\edunet_internship\TrashType_Image_Dataset"
IMAGE_SIZE = (64, 64)

# 2. Extract color histogram features from an image
def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, IMAGE_SIZE)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [4, 4, 4], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

# 3. Load dataset and extract features
features, labels = [], []
classes = os.listdir(DATASET_DIR)

for label in classes:
    class_dir = os.path.join(DATASET_DIR, label)
    for file in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file)
        try:
            feature = extract_features(file_path)
            features.append(feature)
            labels.append(label)
        except:
            print(f"Skipping corrupted file: {file_path}")

print(f"Total images processed: {len(features)}")

# 4. Prepare train/test data
X = np.array(features)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train a Support Vector Machine (SVM) classifier
model = SVC(kernel='rbf', gamma='scale', C=10, random_state=42)
model.fit(X_train, y_train)

# 6. Predict and evaluate
y_pred = model.predict(X_test)

print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Optional: Plot confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix - SVM")
plt.show()