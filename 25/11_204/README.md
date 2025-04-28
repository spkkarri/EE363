#  Battery SOH & RUL Prediction using Machine Learning Models

## 🔬 Overview
This project aims to predict the **State of Health (SOH)** and **Remaining Useful Life (RUL)** of lithium-ion batteries using deep learning models. It utilizes NASA’s battery datasets and features an interactive web interface built using **Flask** **pytorch**  . The core of the system consists of multiple deep learning models designed to analyze degradation patterns and forecast battery health.

## 🧠 Core Machine Learning Models

### 🔸 1. LSTM-CNN
- Hybrid model combining Long Short-Term Memory (LSTM) with Convolutional Neural Networks (CNN).
- Captures sequential dependencies and local patterns in battery degradation.

### 🔸 2. Transformer
- Leverages self-attention mechanisms to model global dependencies.
- Uses self-attention mechanisms to model global dependencies.
- Provides superior performance on long sequences.

## 📈 Common Features Across Models
- **Input:** 5 previous SOH values.
- **Output:** Predicted SOH and RUL.
- **Evaluation Metrics:** MAE (Mean Absolute Error), RMSE (Root Mean Square Error).

## 📁 Dataset
- **Source:** NASA Prognostics Data Repository  
- **Files:** `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0008.mat`
- **Features:** Capacity, charge cycles, temperature, integrated current, etc.

### 🔹 SOH Calculation

```
SOH = (Ct / Cinitial) × 100
```

---

## 📂 Project Structure

# 📂 Project Structure

```
Code/ 
├── static/            # Generated plots (SOH, RUL) and outputs
├── models.py           # Jupyter notebooks for experimentation 
├── data/           # NASA .mat battery datasets extracted from bash command or ps4
│   └── *.mat
├── templates/           #frontend
│   └── index.htmnl
├── uploads/            # Uploaded dataset storage
│   └── *.mat
├── app.py
├── download_data.sh
├── fetchcode_of_EE_363.sh   #extract all files of EE363 from github
├── download_google_drive_folder.ps1 #downloads the dataset as data folder with data
├── utils.py
├── requirements.txt 
└── README.md
```

---



## 🚀 Features

- **Dataset Selection:** Choose from NASA datasets or upload your own `.mat` file.
- **Model Selection:** Select from LSTM-CNN or Transformer.
- **Visualizations:**
  - Predicted vs Actual SOH plot
  - RUL over cycles
  - SOH heatmap and histogram
  - Cycle-wise prediction table
- **Metrics Displayed:** MAE, RMSE for model evaluation.



### 1. Clone this repository

```bash
git clone https://github.com/spkkarri/EE363.git
cd EE363/25/11_204/code

or install all the files in same order

```

### 2. Install dataset
enter the below bash command to fetch the data set
```bash

sh download_data.sh





```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Project 

```bash
python app.py


Visit the app in your browser at [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
```




## 📈 Results and Metrics

- **Predicted vs Actual SOH**
- **RUL Predictions**
- **SOH Heatmap**
- **Cycle-wise Table**
- **MAE and RMSE metrics**

## 🛠 Technologies Used

| Layer         | Tools / Frameworks                     |
|---------------|-----------------------------------------|
| Frontend      | Flask, HTML/CSS                        |
| Backend       | Flask                        |
| ML Models     | PyTorch / TensorFlow (as applicable)   |
| Visualization | Matplotlib, Seaborn                    |
| Data          | NASA Prognostics Data Repository       |

## 📽 Demo
🎥 Watch the full demo here: [*https://drive.google.com/file/d/1nBCN-WwMf0GaRWh8786dTDX6URa8MeGb/view?usp=sharing*]

## DATA SET DRIVE LINK
Link:https://drive.google.com/drive/folders/1inPsAmxpGm4bOLWiutJG9nv7APLpqZGy


### 👨‍💻 Team Information

- **Team No:** 11  
- **Team Lead Roll Number (Last 3 Digits):** 204  
- **Course:** EE363 – Machine Learning for Engineers

**Team Members:**
- `522204` – MATTE BABI SNEHITH KUMAR 
- `522247` – THIRUMALESH UPPARA
- `522106` – ARADI INDRA KUMAR
- `522238` – SLN SWAMY  
- `522149` – KHUSHAL
                                   

## 📄 License
This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.
Video link:
https://drive.google.com/file/d/1nBCN-WwMf0GaRWh8786dTDX6URa8MeGb/view?usp=sharing


