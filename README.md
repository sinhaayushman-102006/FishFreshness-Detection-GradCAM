# 🐟 Fish Freshness Detection AI (FreshSense-AI)

---

## 🚀 Introduction

**FreshSense-AI** is a deep learning-powered web application that detects the freshness of fish using image classification and real-time camera input. It leverages **Computer Vision + Explainable AI (Grad-CAM)** to provide accurate and interpretable predictions.

The system classifies fish into:

* ✅ Fresh
* ⚠️ Medium
* ❌ Spoiled

---

## 💡 Problem Statement

Traditional fish freshness detection:

* Depends on human experience 👤
* Is inconsistent ❌
* Cannot scale efficiently 📉

👉 This project solves it using **AI-based automated classification**.

---

## 🌟 Key Features

### 🖼️ Image Upload

* Upload fish images
* Get prediction + confidence score

### 🎥 Live Camera Detection

* Real-time webcam classification
* Continuous prediction overlay

### 🔥 Grad-CAM (Explainable AI)

* Highlights important regions in image
* Improves transparency and trust

### 🎨 Modern UI

* Dark-themed responsive interface
* Clean card-based layout

---

## 🧠 Innovation & Uniqueness

✔ Combines **Deep Learning + Explainable AI**
✔ Real-time + static image prediction
✔ Practical application in food safety
✔ Lightweight and deployable Flask app

---

## 🌍 Applications

* 🐟 Fish markets
* 🏭 Seafood industries
* 🚚 Supply chain monitoring
* 🧪 Food quality research
* 🛒 Smart retail systems

---

## 🛠️ Tech Stack

| Category      | Technology               |
| ------------- | ------------------------ |
| Backend       | Flask (Python)           |
| AI Model      | PyTorch (EfficientNetV2) |
| CV Processing | OpenCV                   |
| Frontend      | HTML, CSS                |
| XAI           | Grad-CAM                 |
| Deployment    | Localhost / Cloud        |

---

## 📁 Project Structure

```
JU_HACKATHON/
│
├── dataset/
│   ├── train/
│   │   ├── fresh/
│   │   ├── medium/
│   │   └── spoiled/
│   │
│   └── val/
│       ├── fresh/
│       ├── medium/
│       └── spoiled/
│
├── model/                 # (Optional saved models / checkpoints)
│
├── scripts/               # Utility or helper scripts (if any)
│
├── static/
│   ├── style.css          # UI styling
│   └── (uploaded images + Grad-CAM outputs)
│
├── templates/
│   ├── index.html         # Main UI
│   └── camera.html        # Camera UI (if separated)
│
├── venv/                  # Virtual environment
│
├── app.py                 # Main Flask app
├── model.py               # Model architecture definition
├── train.py               # Training script
├── utils.py               # Helper functions
├── gradcam.py             # Grad-CAM implementation
├── export_onnx.py         # Model export (optional)
├── best_model.pth         # Trained model weights
├── requirements.txt       # Dependencies
│
└── README.md              # Documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/freshsense-ai.git
cd freshsense-ai
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

---

### 3️⃣ Activate Environment

```bash
venv\Scripts\activate
```

---

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

*(If requirements.txt not used)*

```bash
pip install flask torch torchvision opencv-python pillow
```

---

## 🧠 Model Training (Optional)

If you want to retrain the model:

```bash
python train.py
```

👉 Uses dataset from:

```
dataset/train
dataset/val
```

---

## ▶️ Run the Application

```bash
python app.py
```

---

## 🌐 Open in Browser

```
http://127.0.0.1:5000
```

---

## 📊 How It Works

1. User uploads image / opens camera
2. Image is preprocessed (resize, tensor conversion)
3. Model predicts freshness class
4. Confidence score is calculated
5. Grad-CAM generates heatmap
6. Results displayed in UI

---

## 📸 Output

* ✔ Prediction: Fresh / Medium / Spoiled
* ✔ Confidence Score (%)
* ✔ Grad-CAM Visualization
* ✔ Live Camera Detection

---

## 🔮 Future Enhancements

* 📱 Mobile App
* ☁️ Cloud Deployment (AWS / GCP)
* 📦 Multi-food detection
* 🔊 Alert system
* 📊 Analytics dashboard

---

## 👩‍💻 Author

**Debasree Sinha**

---

## 🏆 Hackathon Highlights

✔ Real-world problem solving
✔ AI + Explainability combined
✔ Clean UI + Working prototype
✔ Easy demo (Upload + Camera)

---

## ⭐ Conclusion

This project shows how **AI can revolutionize food quality analysis** by making it:

* Faster ⚡
* Accurate 🎯
* Transparent 🔍

---

## 🙌 Support

If you like this project:

⭐ Star the repo
🍴 Fork it
🚀 Build upon it

---
