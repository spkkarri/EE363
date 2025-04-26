# 🌍 SegFormer-B3 Based Semantic Segmentation Project

## 👥 Contributors
- 🎓 *Jeevana Sri* - 522111  
- 🎓 *G. Chaitanya* - 522128  
- 🎓 *G. Mohana Yuktha* - 522129  
- 🎓 *G. Sushma* - 522132  
- 🎓 *G. Tejaswini* - 522133  
- 🎓 *S. Sai Keerti* - 522161  

---

## 🎥 Demo Video
Check out our demo showcasing how our model performs semantic segmentation:  
📽 [Watch Video](https://drive.google.com/file/d/1BEGwgx0MzWgF-dTlNLxYRhbYYAHDd4Y8/view?usp=sharing)

---

## 🎯 Project Overview
This project focuses on semantic segmentation using the SegFormer-B3 architecture to classify land cover types from satellite imagery. Leveraging the DeepGlobe Land Cover Classification Dataset, the model accurately segments terrain into categories like urban, forest, and agriculture. The project demonstrates real-time segmentation capabilities suitable for practical applications.
---

## 🧠 About SegFormer-B3
**SegFormer**, developed by NVIDIA, is a powerful semantic segmentation model with several architectural innovations:

- 🔹 **Hierarchical Transformer Encoder**: Extracts features at multiple scales using a pyramid structure  
- 🔹 **MLP Decoder**: Lightweight and efficient decoder for fast inference  
- 🔹 **No Positional Encoding**: Enhances adaptability to varying input sizes  
- 🔹 **Scalable Design**: B3 version strikes a balance between performance and computational efficiency  

---

## 🗂 Dataset Used
We used the **DeepGlobe Land Cover Classification Dataset**, which consists of high-resolution satellite images labeled with different land types such as urban, agriculture, and forest.

🔗 [Kaggle - DeepGlobe Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)

---

## 📄 Research Paper
For a deeper understanding of SegFormer’s architecture and theoretical background, refer to the official research paper:  
📑 [SegFormer Research Paper](https://paperswithcode.com/paper/segformer-simple-and-efficient-design-for#code)

---

## 💾 Weights
Use this weights to directly deploy the model(Weights after Finetuning):

📥 [Download Weights](https://drive.google.com/file/d/1oV7jqy3EJDr-pSMEqxWi5d1eVIfpv4W2/view?usp=sharing)

---

## 🚀 How to Run the Project

### Step 1: Clone the Repository
Manually download and run the `clone_Land_Segmentation.sh` file from the repository page to clone the repository.

### Step 2: Install Dependencies
Navigate to the project directory and install required dependencies:
```bash
pip install -r requirements.txt
```

### Step 3: Run the code
```bash
python app.py
