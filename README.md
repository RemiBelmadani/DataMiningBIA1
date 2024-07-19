# Interactive Data Analysis and Clustering Web Application

## Dataset 

parkinsons_updrs.csv

## Link Github

https://github.com/RemiBelmadani/DataMiningBIA1

## Presentation video

https://efrei365net.sharepoint.com/sites/test987/Documents%20partages/General/Recordings/R%C3%A9union%20dans%20%C2%AB%C2%A0General%C2%A0%C2%BB-20240719_160817-Enregistrement%20de%20la%20r%C3%A9union.mp4?web=1&referrer=Teams.TEAMS-WEB&referrerScenario=MeetingChicletGetLink.view

## Description

This project is an interactive web application using Streamlit to analyze, clean and visualize data. The application implement clustering algorithms to group similiar object in the dataset. The application is developed using Streamlit and offers various features for mining and processing data from a file to import.

## Features

- Loading CSV files
- Initial data exploration (overview, statistical summary)
- Data pre-processing and cleaning (management of missing values, normalization)
- Visualization of cleaned data (histograms, box plots)
- Clustering (K-Means, DBSCAN) and visualization of clusters
- Prediction (Logistic Regression, Random Forest) and evaluation of results

## Installing the project

1. **Clone the repository:**
 - Open your terminal (or command prompt).
 - Run the following command to clone the repository:
 ```bash
 git clone <repository-url>
 ```
 - Access the cloned project directory:
 ```bash
 cd <repository-directory>
 ```

2. **Create a virtual environment (optional but recommended):**
 - Create a virtual environment to isolate project dependencies:
 ```bash
 python -m venv env
 ```
 - Activate the virtual environment:
 - On Windows:
 ```bash
 .\env\Scripts\activate
 ```
 - On macOS and Linux:
 ```bash
 source env/bin/activate
 ```

3. **Install dependencies:**
 - Run the following command to install all necessary dependencies listed in the `requirements.txt` file:
 ```bash
 pip install -r requirements.txt
 ```

## Using the app

1. **Launch the application:**
 - Make sure you are in the project directory.
 - Run the following command to launch the Streamlit application:
 ```bash
 streamlit run app.py
 ```

2. **Upload a CSV file:**
 - In the app's web interface, use the sidebar to upload your CSV file.
 - Specify whether the CSV file contains a header and the delimiter used (default `,`).
