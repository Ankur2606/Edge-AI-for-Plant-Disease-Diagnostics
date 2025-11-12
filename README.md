# 🌿 Edge AI for In-Field Crop Disease Diagnostics  
### Creative Problem Solving (Course Code: 2280722)

This repository contains the complete implementation, codebase, and documentation for the project **“Edge AI and On-Device Deep Models for In-Field Diagnostics”**, developed as part of the *Creative Problem Solving* course requirement.

---

## 🧠 Project Overview
Crop disease detection plays a vital role in sustainable agriculture. Farmers often rely on manual inspection, which is time-consuming and error-prone.  
This project leverages **Deep Learning and Edge AI** to automate the identification of tomato leaf diseases using lightweight convolutional models that can be deployed on mobile or edge devices.

We trained and compared three edge-efficient architectures:
- **EfficientNetB0**
- **MobileNetV2**
- **NASNetMobile**

The models were trained on a subset of the **PlantVillage** dataset (Tomato leaves) to evaluate classification performance, inference speed, and deployability on low-power hardware.

---

## 🧩 Problem Statement
> Develop and evaluate multiple lightweight Deep Learning models for **on-device crop disease diagnosis**, focusing on optimizing accuracy, model size, and generalization for edge deployment.

---

## 🎯 Objectives
1. Preprocess and organize the PlantVillage Tomato dataset for multi-class disease classification.  
2. Implement and train three edge-optimized CNN models: EfficientNetB0, MobileNetV2, NASNetMobile.  
3. Evaluate model accuracy, loss, and effectiveness using validation data.  
4. Export the best-performing model to **TensorFlow Lite (TFLite)** for edge deployment.  
5. Provide analytical visualization comparing model effectiveness.

---

## 🧮 Dataset
**Source:** [PlantVillage Tomato Leaf Dataset – Kaggle](https://www.kaggle.com/datasets/charuchaudhry/plantvillage-tomato-leaf-dataset/code/data)
**Classes Used:**  
`Tomato___Bacterial_spot`, `Tomato___Early_blight`, `Tomato___Late_blight`, `Tomato___Leaf_Mold`, `Tomato___Septoria_leaf_spot`, `Tomato___Spider_mites Two-spotted_spider_mite`, `Tomato___Target_Spot`, `Tomato___Tomato_Yellow_Leaf_Curl_Virus`, `Tomato___Tomato_mosaic_virus`, `Tomato___healthy`.  
Each image (224×224 px RGB) was resized, augmented, and normalized for model input.

---

## 🏗️ System Architecture
```

[Input Images] → [Preprocessing & Augmentation]
→ [Model (EfficientNet / MobileNet / NASNet)]
→ [Prediction Layer (Softmax)]
→ [Classification Output]
→ [Export to TFLite for Edge Deployment]

````

---

## ⚙️ Installation & Usage

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/Ankur2606/Edge-AI-for-Plant-Disease-Diagnostics.git
cd Edge-AI-for-Plant-Disease-Diagnostics
````

### **2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Notebook (Kaggle / Colab / Local)**

```bash
jupyter notebook mobile-net-edge-prediction.ipynb
```

Or directly open in Kaggle with GPU enabled.

### **4️⃣ Model Training**

Training automatically:

* Splits dataset (80-20)
* Trains all three architectures
* Logs validation accuracy & loss
* Saves best checkpoints in `/working/`

### **5️⃣ Model Comparison & Results**

After training, run:

```python
# Display comparison summary and graphs
print(summary_df)
```

Generates:

* Validation Accuracy & Loss summary
* Effectiveness plots
* Confusion matrices per model

### **6️⃣ Export Best Model to TFLite**

```python
# Converts and saves quantized TFLite model
converter = tf.lite.TFLiteConverter.from_saved_model('/kaggle/working/EfficientNetB0_saved_model')
tflite_model = converter.convert()
```

Resulting file:
`/kaggle/working/EfficientNetB0_float32.tflite`

---

## 🧪 Results Summary

| Model              | Validation Accuracy | Effectiveness |    Relative Improvement   |
| :----------------- | :-----------------: | :-----------: | :-----------------------: |
| **EfficientNetB0** |     **86.73 %**     |   **0.558**   | **+5.9 % vs MobileNetV2** |
| MobileNetV2        |       81.90 %       |     0.458     |             —             |
| NASNetMobile       |       77.14 %       |     0.271     |             —             |

✅ **Best model:** *EfficientNetB0* – Highest accuracy, stable loss, and best trade-off for edge devices.

---

## 📊 Visualization Outputs

* Confusion matrices for each model
* Accuracy vs (1 − Normalized Loss) chart
* Validation accuracy comparison bar plot
* Effectiveness metric annotation

---

## 📱 Deployment

The exported `model.tflite` can be integrated into:

* **Android** (TensorFlow Lite Interpreter)
* **Raspberry Pi / Jetson Nano**
* **Streamlit / Gradio web apps**

Example Python inference:

```python
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path="EfficientNetB0_float32.tflite")
interpreter.allocate_tensors()
```

---

## 🧾 Repository Structure

```
Edge-AI-Crop-Diagnostics/
│
├── edge_ai_infield_diagnostics.ipynb   # Main training notebook
├── requirements.txt                    # Dependencies
├── README.md                           # Documentation
├── saved_models/                       # Trained Keras models
├── model_float32.tflite                # TFLite model (EfficientNetB0)
├── results/                            # Graphs, metrics, confusion matrices
└── report.pdf                          # Academic report submission
```

---

## 📚 References

1. Howard et al., *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*, arXiv 2017.
2. Tan & Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019.
3. Zoph et al., *Learning Transferable Architectures for Scalable Image Recognition (NASNet)*, CVPR 2018.
4. Hughes et al., *PlantVillage: A Public Dataset for Plant Disease Classification*, arXiv 2015.
5. TensorFlow Documentation (2025). *Keras Applications API Reference*.
6. Kaggle PlantVillage Dataset. *Tomato Leaf Diseases*.
7. TF Lite Guide (2025). *Deploy models on edge devices*.
8. Chollet F. et al., *Deep Learning with Python (2e)*, Manning 2021.
9. Zhang et al. (2024). *Comparative Analysis of Lightweight CNNs for On-Device Agriculture AI*, Computers & Electronics in Agriculture.
10. Google AI Edge Team (2025). *Quantization and Model Optimization Techniques for TensorFlow Lite*.

---

## 🧑‍💻 Team Members

| Name                          | Role                                                        |
| :---------------------------- | :---------------------------------------------------------- |
| **Bhavya Pratap Singh Tomar** | Model Development, Comparative Analysis, Report Preparation |
| **Bhavya Madan**              | Dataset Curation, Preprocessing, Visualization              |
| **Barkha Rai**                | Evaluation Metrics, Documentation, Presentation             |

---


## 📢 License

This project is released for educational and research purposes under the MIT License.

