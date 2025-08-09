````markdown
# 🧠 MRI Brain Tumor Analysis App

This project is a **Streamlit web application** for **brain tumor MRI image analysis**.  
It contains two main functionalities:  
1. **Segmentation** – Detect and highlight tumor regions in MRI scans.  
2. **Classification** – Predict tumor type from MRI images.

---

## 🚀 Features
- **Interactive Web App** built with Streamlit.
- **MRI Image Upload** with a clean UI.
- **Tumor Segmentation** using a trained deep learning model.
- **Tumor Classification** into categories (e.g., glioma, meningioma, pituitary tumor).
- **Download Segmentation Masks** as images.
- **Responsive Design** with custom CSS styling.

---
## 📂 Project Structure
📦 Project
│── app.py # Main application code
│── animations.js # JavaScript animations
│── requirements.txt # Required dependencies
│── README.md # Project documentation
│
├── test_photos/  
├── predictions/ 
├── Segmentation/ 
│ └── brain-tumor-segmentation-using-unet.ipynb 
├── Classification/ 
│ └── brain-tumor-classification.ipynb 

---

## 🚀 How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
````

### 2️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🖼️ How to Use

1. Upload an MRI image via the web interface.
2. Select one of the modes:

   * **Classification** → Predict the tumor type or absence of tumor.
   * **Segmentation** → Highlight the tumor region in the MRI.
3. View the results in the application. Output files will be saved in the `predictions/` folder.

---

## 📌 Requirements

* Python 3.11 or higher
* Main Python libraries:

  * `streamlit`
  * `tensorflow`
  * `numpy`
  * `opencv-python`
  * `Pillow`
  * *(Any other libraries listed in `requirements.txt`)*

---

## 📊 Models Used

* **Classification Models**:

  * ResNet
  * EfficientNet
  * GoogleNet
* **Segmentation Model**:

  * U-Net

---

## 📜 Notes

* You can change folder paths in `app.py` to use other datasets.
* Models were trained on the **Brain MRI segmentation Dataset**.

---

## 📦 Pre-trained Models

Due to file size limitations on GitHub, the trained models are hosted on Google Drive.  
Click the link below to download them:

🔗 [Google Drive - Brain Tumor Models](https://drive.google.com/drive/folders/1c7QqNMkogn2zRylDNzwlzGxPXvBX-BT7?usp=sharing)

---


## 👨‍💻 Author

Developed by **Aya Khaled Farouk**.

```

