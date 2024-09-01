# Cervical-Cancer-Risk-Prediction

### Overview:
This GitHub repository contains code for predicting cervical cancer risk using machine learning techniques. The code encompasses data preprocessing, exploratory data analysis, model building, and evaluation. The dataset used includes various features related to individuals' demographics and medical history to predict the risk of cervical cancer.

### Objective:
The primary objective of this project is to develop a predictive model for cervical cancer risk assessment utilizing machine learning techniques. By analyzing various features such as age, sexual behavior, medical history, and lifestyle factors, the aim is to accurately classify individuals into categories based on their risk of developing cervical cancer. The ultimate goal is to provide early detection and intervention strategies to improve overall health outcomes.

### Problem Statement:
The problem statement is the need to effectively predict and classify individuals into cervical cancer risk categories based on key features present in the dataset. By leveraging machine learning algorithms and predictive modeling, the challenge is to create a robust and accurate model that can assist in identifying individuals who may be at higher risk of cervical cancer. The focus is on optimizing the classification performance to enable timely interventions and personalized healthcare strategies for better patient care and outcomes.

## Repository Structure

### Code Files
- **`cervical_cancer_prediction.ipynb`**: Jupyter Notebook containing the Python code for cervical cancer risk prediction.

### Dataset
- **`cervical_cancer.csv`**: Dataset used for training and testing the predictive models.

## Prerequisites

Ensure you have the following installed:

- **Python**: Version 3.8 or higher
- **Pandas**: Version 1.5.3 or higher
- **NumPy**: Version 1.24.0 or higher
- **Plotly**: Version 5.10.0 or higher
- **Scikit-learn**: Version 1.2.2 or higher
- **Imbalanced-learn**: Version 0.11.0 or higher

## Setup and Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Deepshikha-Chhaperia/Cervical-Cancer-Risk-Prediction.git
   cd Cervical-Cancer-Risk-Prediction
    ```

## Setup and Installation

1. **Install Dependencies**

   Use `pip` to install the required libraries:

   ```bash
   pip install pandas numpy plotly scikit-learn imbalanced-learn
   ```

   ## Prepare the Data

Place the `cervical_cancer.csv` file in the root directory of the repository.

## Instructions to Run the Code

1. **Open the Jupyter Notebook**

   Launch Jupyter Notebook:

   ```bash
   jupyter notebook cervical_cancer_prediction.ipynb
     ```


   ## Execute the Cells

Run each cell sequentially in the notebook to perform data preprocessing, exploratory data analysis, and model building.

## Results

The project provides the following outcomes:

- **Data Analysis**:
  - Insights into contraceptive usage and its distribution across different age groups.
  - Visualization of key features affecting cervical cancer risk.

- **Model Performance**:
  - **Logistic Regression**: Achieved approximately 84% accuracy on the balanced dataset.
  - **K-Nearest Neighbors (KNN)**: Achieved approximately 80% accuracy.
  - **ADASYN**: Utilized for class balancing, enhancing model sensitivity and specificity.

Evaluation metrics such as accuracy, precision, recall, and F1 score were used to assess model performance, demonstrating the effectiveness of the models in predicting cervical cancer risk.


