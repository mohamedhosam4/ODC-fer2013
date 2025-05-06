
# Facial Expression Recognition (FER2013) - Deep Learning Project

This project is a Facial Expression Recognition (FER) model trained on the FER2013 dataset. It was developed as part of the **AI and Data Science Scholarship** from **Orange Digital Center** in collaboration with **Amit Learning**.

The model classifies facial expressions into 7 categories using a CNN-based architecture enhanced by the pre-trained **VGG16** network.

🔗 **Live Web App:** [Click here to try the app!](https://odc-fer2013.streamlit.app/)

---

## 📸 Screenshots

<img src="https://github.com/mohamedhosam4/ODC-fer2013/blob/main/Screenshot%202025-05-06%20181113.png" width="600"/>

<img src="https://github.com/mohamedhosam4/ODC-fer2013/blob/main/Screenshot%202025-05-06%20020144.png" width="600"/>

---

## 🧠 Model Architecture

The model uses the following pipeline:

- Input: 48x48 grayscale facial image
- Convert grayscale to RGB using 1x1 convolution
- Pass to pre-trained **VGG16** (excluding top layers)
- Global Average Pooling to reduce dimensionality
- Dense layers with Batch Normalization and Dropout for generalization
- Output: Softmax over 7 emotion classes

---

## 🧪 Technologies Used

- **Python**
- **TensorFlow / Keras**
- **VGG16 Pre-trained Model**
- **Streamlit** – for deploying the web app
- **FER2013 dataset**

---

## 🎓 Scholarship Info

This project was completed as part of the:

> **AI and Data Science Scholarship**  
> **Orange Digital Center** in collaboration with **Amit Learning**

---

## 🚀 How to Run Locally

1. Clone the repo:
```bash
git clone https://github.com/mohamedhosam4/ODC-fer2013.git
cd ODC-fer2013
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

---

## 🙌 Author

**Mohamed Hosam**  
[GitHub Profile](https://github.com/mohamedhosam4)

---

## 📄 License

This project is licensed for educational purposes only.
