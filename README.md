# Titanic Dataset — Exploratory Data Analysis (EDA)

This repository contains Python scripts to perform exploratory data analysis (EDA) on the Titanic dataset. The goal is to understand the data using summary statistics, visualizations, and basic inferences to identify patterns, trends, and anomalies before any modeling.

---

## Project Overview

EDA is a critical step in any data science workflow. It helps uncover insights, detect anomalies, and guide feature engineering or model selection. This project showcases EDA techniques on the Titanic dataset using Pandas, Matplotlib, and Seaborn.

---

## Step-by-Step Summary

### Step 1: Load Cleaned Data
- Load the cleaned dataset `cleaned_data.csv`.

### Step 2: Summary Statistics
- Generate descriptive statistics (mean, median, std, count) for numerical and categorical variables.
- Check for missing values and unique value counts.

### Step 3: Visualizations
- Create histograms and boxplots for all numerical features to understand distributions and identify outliers.
- Generate a correlation matrix heatmap to explore relationships between numerical features.
- Create pairplots (scatterplot matrix) with kernel density estimates on the diagonal for detailed feature interaction visualization.

### Step 4: Pattern and Anomaly Detection
- Visually inspect plots for skewness, outliers, or unusual distributions.
- Look for strong correlations or lack thereof between features.
- Identify features that show distinct behavior relative to the target variable (`Survived`).

### Step 5: Basic Inferences
- Analyze survival rate differences across gender (`Sex`).
- Examine correlations of features like `Fare` and `Age` with survival.
- Note important observations to guide future modeling or feature selection.

---

## Dependencies

Make sure you have the following Python libraries installed:

- pandas  
- matplotlib  
- seaborn  

Install them via pip:

```bash
pip install pandas matplotlib seaborn
