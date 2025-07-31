# ♻️ Garbage Classification Web App

This project is a simple and interactive web application for classifying garbage images into six categories using a machine learning model (SVM) and deployed using Streamlit.

> 🔍 Built as part of the **AICTE – Edunet Foundation Green Skills Internship** (June–July 2025)

---

## 🧠 Project Overview

The goal of this project is to classify waste images into one of the following six classes:

- **cardboard**
- **glass**
- **metal**
- **paper**
- **plastic**
- **trash**

The model is trained on color histogram features extracted from the images and uses a Support Vector Machine (SVM) classifier for prediction.

---


*(Output screenshots are available in the `output_screenshots/` folder)*

---

## 🚀 Features

- 📤 Upload your own garbage image
- ⚡ Get an instant predicted class
- ✅ Clean and minimal UI with helpful alerts

---

## ⚠️ Limitations

This is a **basic machine learning model** trained only on a fixed dataset. It may incorrectly classify unrelated images (like human faces, nature, etc.) as one of the garbage categories. The app is a proof of concept built during an internship project.

---

## 📦 Installation & Setup

To run this project locally, follow these steps:

```bash
# 1. Clone the repository
git clone https://github.com/Rushmitha1422/garbage_classification.git
cd garbage_classification

# 2. Install required packages
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

---

## 📁 Project Structure 
```
.
├── app.py # Streamlit web app
├── final.py # SVM model training script
├── svm_model.pkl # Saved trained model
├── output_screenshots/ # Screenshots from app output
├── TrashType_Image_Dataset/ # Dataset (for training only)
├── presentation.pptx # Final project PPT (internship)
└── README.md # This file 
```

---

## 🧠 Model Training Summary

The model is trained in the `final.py` file using the following steps:

- ✅ Loads images from six classes: cardboard, glass, metal, paper, plastic, and trash  
- ✅ Resizes and converts images to HSV color space  
- ✅ Extracts **color histogram features** (4×4×4 = 64 bins)  
- ✅ Trains a **Support Vector Machine (SVM)** with RBF kernel  
- ✅ Saves the trained model to `svm_model.pkl`  
- ✅ Evaluates performance with accuracy, classification report, and confusion matrix

This model is used by the Streamlit app to classify uploaded images in real-time.

---

##  Acknowledgements

This project was completed as part of a 4-week **virtual internship** on  
**Artificial Intelligence and Data Analytics with Green Skills**, organized by:

- **AICTE**
- **Edunet Foundation**
- **Shell India Markets Pvt. Ltd.**

📅 Internship Duration: 16th June 2025 – 16th July 2025  
📌 Internship Theme: AI for Green & Sustainable Development (Skills4Future)

---

## 📬 Contact

Feel free to explore, fork, or star this project.

🔗 GitHub: [github.com/Rushmitha1422](https://github.com/Rushmitha1422)  
📫 LinkedIn: [linkedin.com/in/rushmitha-ubbara](https://www.linkedin.com/in/rushmithareddyubbara)  

## 🚀 Live Demo

Try out the Garbage Classification Streamlit App here:  
👉 [Click to Launch App](https://garbage-classification-tool.streamlit.app/)

---



