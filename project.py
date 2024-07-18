import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Interactive Data Analysis and Clustering Web Application")

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
    st.write("First 5 rows:")
    st.dataframe(df.head())

    st.write("Last 5 rows:")
    st.dataframe(df.tail())

    # Statistical Summary
    st.subheader("Statistical Summary")

    # Number of lines and columns
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")

    # Column names
    st.write("Column Names:")
    st.write(df.columns.tolist())

    # Missing values per column
    st.write("Missing Values Per Column:")
    st.write(df.isnull().sum())

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

    task = st.selectbox("Choose a task", ["None", "Clustering", "Prediction"])

    if task == "Clustering":
        clustering_method = st.selectbox("Choose a clustering method", ["None", "K-Means", "DBSCAN"])

        if clustering_method == "K-Means":
            num_clusters = st.number_input("Number of clusters (k)", min_value=1, max_value=10, value=3)
            kmeans = KMeans(n_clusters=num_clusters)
            clusters = kmeans.fit_predict(df)
            df['Cluster'] = clusters
            st.write("K-Means clustering completed")
        elif clustering_method == "DBSCAN":
            eps_value = st.number_input("Epsilon (eps)", min_value=0.1, max_value=10.0, value=0.5)
            min_samples_value = st.number_input("Minimum samples", min_value=1, max_value=10, value=5)
            dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
            clusters = dbscan.fit_predict(df)
            df['Cluster'] = clusters
            st.write("DBSCAN clustering completed")

        st.write("Data with clusters:")
        st.dataframe(df.head())

        # Visualization of clusters
        st.subheader("Cluster Visualization")

        pca = PCA(2)
        pca_result = pca.fit_transform(df.drop(columns=['Cluster']))
        df['PCA1'] = pca_result[:, 0]
        df['PCA2'] = pca_result[:, 1]

        fig, ax = plt.subplots()
        scatter = ax.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'], cmap='viridis')
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)
        st.pyplot(fig)

        # Cluster statistics
        st.subheader("Cluster Statistics")

        cluster_stats = df.groupby('Cluster').size().reset_index(name='Count')
        st.write(cluster_stats)

    elif task == "Prediction":
        prediction_method = st.selectbox("Choose a prediction method", ["None", "Logistic Regression", "Random Forest"])

        target_column = st.selectbox("Select the target column", df.columns)
        if target_column:
            X = df.drop(columns=[target_column])
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if prediction_method == "Logistic Regression":
                lr = LogisticRegression()
                lr.fit(X_train, y_train)
                predictions = lr.predict(X_test)
                st.write("Logistic Regression completed")
            elif prediction_method == "Random Forest":
                rf = RandomForestClassifier()
                rf.fit(X_train, y_train)
                predictions = rf.predict(X_test)
                st.write("Random Forest completed")

            st.write("Classification Report:")
            st.text(classification_report(y_test, predictions))

    
