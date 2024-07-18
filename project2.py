import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Interactive Data Analysis and Clustering Web Application")

# Part 1
# Data loading
st.sidebar.header("Data Loading Options")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    header_option = st.sidebar.checkbox("Does your CSV have a header?", value=True)
    delimiter_option = st.sidebar.text_input("Delimiter", value=",")

    if header_option:
        df = pd.read_csv(uploaded_file, delimiter=delimiter_option)
    else:
        df = pd.read_csv(uploaded_file, delimiter=delimiter_option, header=None)

    st.write("Data Loaded Successfully")

    # Data Preview
    st.subheader("Data Preview")
    num_rows = st.number_input("Number of rows to view", min_value=1, value=5)
    st.write(f"First {num_rows} rows:")
    st.dataframe(df.head(num_rows))

    st.write(f"Last {num_rows} rows:")
    st.dataframe(df.tail(num_rows))

    # Statistical Summary
    st.subheader("Statistical Summary")

    # Number of rows and columns
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")

    # Column names
    st.write("Column Names:")
    st.write(df.columns.tolist())

    # Data types
    st.write("Data Types:")
    st.write(df.dtypes)

    # Check for missing values
    st.write("Missing Values Per Column:")
    st.write(df.isnull().sum())

    # Check for duplicate rows
    st.write("Number of Duplicate Rows:")
    st.write(df.duplicated().sum())

    # Basic statistics
    st.subheader("Basic Statistics")
    st.write(df.describe())

    # Handle missing values
    if st.sidebar.button("Handle Missing Values"):
        df = df.fillna(df.mean())
        st.write("Missing values filled with column mean.")
        st.write(df.isnull().sum())


    # Part 2
    # Data Pre-processing and Cleaning
    st.subheader("Data Pre-processing and Cleaning")

    # Handling Missing Values
    st.write("Handling Missing Values")
    missing_value_method = st.selectbox("Choose a method to handle missing values", 
                                        ["None", "Drop Rows", "Drop Columns", "Replace with Mean", "Replace with Median", "Replace with Mode", "KNN Imputer"])

    if missing_value_method == "Drop Rows":
        df = df.dropna()
    elif missing_value_method == "Drop Columns":
        df = df.dropna(axis=1)
    elif missing_value_method == "Replace with Mean":
        imputer = SimpleImputer(strategy='mean')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    elif missing_value_method == "Replace with Median":
        imputer = SimpleImputer(strategy='median')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    elif missing_value_method == "Replace with Mode":
        imputer = SimpleImputer(strategy='most_frequent')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    elif missing_value_method == "KNN Imputer":
        imputer = KNNImputer()
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    st.write("Data after handling missing values:")
    st.dataframe(df.head())

    # Data Normalization
    st.write("Data Normalization")
    normalization_method = st.selectbox("Choose a normalization method", 
                                        ["None", "Min-Max Normalization", "Z-score Standardization"])

    if normalization_method == "Min-Max Normalization":
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    elif normalization_method == "Z-score Standardization":
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    st.write("Data after normalization:")
    st.dataframe(df.head())

    # Part III: Visualization of the cleaned data
    st.subheader("Visualization of the Cleaned Data")

    # Histograms
    st.write("Histograms")
    column_to_plot = st.selectbox("Select column for histogram", df.columns)
    fig, ax = plt.subplots()
    ax.hist(df[column_to_plot], bins=30, edgecolor='black')
    plt.title(f'Histogram of {column_to_plot}')
    plt.xlabel(column_to_plot)
    plt.ylabel('Frequency')
    st.pyplot(fig)

    # Box plots
    st.write("Box Plots")
    column_to_plot = st.selectbox("Select column for box plot", df.columns, key='box')
    fig, ax = plt.subplots()
    ax.boxplot(df[column_to_plot], vert=False)
    plt.title(f'Box Plot of {column_to_plot}')
    plt.xlabel(column_to_plot)
    st.pyplot(fig)


    # Part IV: Clustering or Prediction
    st.subheader("Clustering or Prediction")

    task = st.selectbox("Choose a task", ["None", "Clustering", "Prediction"])

    if task == "Clustering":
        clustering_method = st.selectbox("Choose a clustering method", ["None", "K-Means", "DBSCAN", "Agglomerative Clustering"])

        if clustering_method == "K-Means":
            num_clusters = st.slider("Number of clusters (k)", min_value=1, max_value=10, value=3)  # Slider to choose the number of clusters
            kmeans = KMeans(n_clusters=num_clusters)
            clusters = kmeans.fit_predict(df)
            df['Cluster'] = clusters
            st.write("K-Means clustering completed")

            # Visualization of clusters
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df.drop(columns=['Cluster']))
            pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
            pca_df['Cluster'] = df['Cluster']

            st.subheader("PCA Projection of Clusters")
            fig, ax = plt.subplots()
            scatter = ax.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['Cluster'], cmap='viridis', s=50)
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.title('PCA Projection of Clusters')
            legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend1)
            st.pyplot(fig)

            # Cluster statistics
            st.subheader("Cluster Statistics")
            cluster_centers = kmeans.cluster_centers_
            for i in range(num_clusters):
                st.write(f"Cluster {i}:")
                st.write(f"Center: {cluster_centers[i]}")
                st.write(f"Number of points: {np.sum(clusters == i)}")

        elif clustering_method == "DBSCAN":
            eps_value = st.number_input("Epsilon (eps)", min_value=0.1, max_value=10.0, value=0.5)
            min_samples_value = st.number_input("Minimum samples", min_value=1, max_value=10, value=5)
            dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
            clusters = dbscan.fit_predict(df)
            df['Cluster'] = clusters
            st.write("DBSCAN clustering completed")

            # Visualization of clusters
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df.drop(columns=['Cluster']))
            pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
            pca_df['Cluster'] = df['Cluster']

            st.subheader("PCA Projection of Clusters")
            fig, ax = plt.subplots()
            scatter = ax.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['Cluster'], cmap='viridis', s=50)
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.title('PCA Projection of Clusters')
            legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend1)
            st.pyplot(fig)

            # Cluster statistics
            st.subheader("Cluster Statistics")
            unique_clusters = np.unique(clusters)
            for cluster in unique_clusters:
                if cluster != -1:
                    st.write(f"Cluster {cluster}:")
                    st.write(f"Number of points: {np.sum(clusters == cluster)}")
                    st.write(f"Density: {np.sum(clusters == cluster) / len(df)}")
                else:
                    st.write(f"Noise points: {np.sum(clusters == cluster)}")

        elif clustering_method == "Agglomerative Clustering":
            num_clusters = st.slider("Number of clusters (k)", min_value=1, max_value=10, value=3)  # Slider to choose the number of clusters
            linkage_method = st.selectbox("Linkage method", ["ward", "complete", "average", "single"])
            agglomerative = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage_method)
            clusters = agglomerative.fit_predict(df)
            df['Cluster'] = clusters
            st.write("Agglomerative Clustering completed")

            # Visualization of clusters
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df.drop(columns=['Cluster']))
            pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
            pca_df['Cluster'] = df['Cluster']

            st.subheader("PCA Projection of Clusters")
            fig, ax = plt.subplots()
            scatter = ax.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['Cluster'], cmap='viridis', s=50)
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.title('PCA Projection of Clusters')
            legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend1)
            st.pyplot(fig)

            # Cluster statistics
            st.subheader("Cluster Statistics")
            for i in range(num_clusters):
                st.write(f"Cluster {i}:")
                st.write(f"Number of points: {np.sum(clusters == i)}")

        st.write("Data with clusters:")
        st.dataframe(df.head())

    elif task == "Prediction":
        prediction_method = st.selectbox("Choose a prediction method", ["None", "Random Forest", "Linear Regression"])

        if prediction_method == "Random Forest":
            target_column = st.selectbox("Select target column for classification", df.columns)
            X = df.drop(columns=[target_column])
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            n_estimators = st.number_input("Number of estimators", min_value=10, max_value=100, value=50)
            rf = RandomForestClassifier(n_estimators=n_estimators)
            rf.fit(X_train, y_train)
            predictions = rf.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            st.write(f"Random Forest Classifier Accuracy: {accuracy}")

        elif prediction_method == "Linear Regression":
            target_column = st.selectbox("Select target column for regression", df.columns)
            X = df.drop(columns=[target_column])
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            predictions = lr.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            st.write(f"Linear Regression Mean Squared Error: {mse}")

        # Ensure Cluster column is dropped before performing PCA
        if 'Cluster' in df.columns:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df.drop(columns=['Cluster']))
        else:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df)
        
        pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
        pca_df['Cluster'] = df['Cluster']

        st.subheader("PCA Projection of Clusters")
        fig, ax = plt.subplots()
        scatter = ax.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['Cluster'], cmap='viridis', s=50)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('PCA Projection of Clusters')
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        st.pyplot(fig)