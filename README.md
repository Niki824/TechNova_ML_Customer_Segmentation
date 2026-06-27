# TechNova Customer Segmentation using K-Means Clustering

## Overview

This project was developed as part of the TechNova Machine Learning Internship Task 3: Clustering with K-Means.

The objective of this project is to segment customers into different groups based on their annual income and spending behavior using the K-Means clustering algorithm.

## Features

* Data loading and preprocessing
* Customer segmentation using K-Means clustering
* Determination of optimal number of clusters using the Elbow Method
* Visualization of customer groups
* Cluster centroid visualization

## Dataset

Dataset: Mall Customers Dataset

Dataset Features:

* CustomerID
* Gender
* Age
* Annual Income (k$)
* Spending Score (1-100)

Features used for clustering:

* Annual Income (k$)
* Spending Score (1-100)

## Methodology

1. Load the dataset.
2. Select relevant features for clustering.
3. Apply the Elbow Method to determine the optimal number of clusters.
4. Train the K-Means clustering model.
5. Visualize customer clusters and centroids.

## Results

* Optimal number of clusters identified: **5**
* Customer segments successfully visualized.
* Elbow method graph generated for cluster selection.

## Generated Files

* `elbow_method.png`
* `customer_clusters.png`

## Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn

## Project Structure

```text
TechNova_Customer_Segmentation/
│
├── customer_segmentation.py
├── Mall_Customers.csv
├── elbow_method.png
├── customer_clusters.png
├── requirements.txt
├── README.md
└── .gitignore
```

## Internship Task

TechNova Machine Learning Internship - Task 3: Clustering with K-Means
