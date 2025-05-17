import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Streamlit app title
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🧠 Advanced Customer Segmentation Tool")
st.markdown("Upload your dataset (CSV format) to perform deep clustering-based segmentation.")

# Sample download
file_path = "customer_segmentation_dataset (2).csv"
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        st.download_button("📥 Download Sample Dataset", f, file_name="customer_segmentation_sample.csv", mime="text/csv")
else:
    st.warning("Sample dataset not found.")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
        st.stop()

    st.write("### 📊 Dataset Preview")
    st.dataframe(df.head())

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Identify column types
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Encode categorical variables
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        st.info(f"Encoded categorical columns: {categorical_cols}")
    else:
        st.info("No categorical columns detected.")

    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # PCA to reduce dimensionality
    pca = PCA()
    df_pca = pca.fit_transform(df)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(explained_variance >= 0.95) + 1
    st.write(f"📉 PCA: Retaining {n_components} components (95% variance explained)")
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df)

    # Build autoencoder
    input_dim = df_pca.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train autoencoder
    split_idx = int(0.8 * len(df_pca))
    X_train, X_val = df_pca[:split_idx], df_pca[split_idx:]
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, validation_data=(X_val, X_val), verbose=0)

    # Get encoded representation
    encoder = Model(input_layer, encoded)
    X_encoded = encoder.predict(df_pca)

    # Choose number of clusters
    n_clusters = st.number_input("Select number of clusters", min_value=2, max_value=10, value=5)

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_encoded)

    # Save models
    autoencoder.save("autoencoder_model.keras")
    pickle.dump(kmeans, open("kmeans_model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))
    pickle.dump(pca, open("pca.pkl", "wb"))

    # Assign clusters
    df['Cluster'] = kmeans.labels_

    st.header("📈 Segmentation Results")

    # Group distribution
    st.subheader("How Many Customers Are in Each Group?")
    cluster_counts = df['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Customer Group', 'Number of Customers']
    fig1 = px.bar(cluster_counts, x='Customer Group', y='Number of Customers', color='Customer Group')
    st.plotly_chart(fig1)

    # 2D visualization
    st.subheader("Customer Groups in 2D Space")
    pca_2d = PCA(n_components=2)
    df_pca_2d = pca_2d.fit_transform(df.drop(columns=['Cluster']))
    df_pca_2d = pd.DataFrame(df_pca_2d, columns=['PC1', 'PC2'])
    df_pca_2d['Cluster'] = df['Cluster']
    fig2 = px.scatter(df_pca_2d, x='PC1', y='PC2', color='Cluster', title="2D Projection of Clusters")
    st.plotly_chart(fig2)

    # Key metrics across groups
    st.subheader("What Makes Each Group Different?")
    feature_importance = df.groupby('Cluster')[numerical_cols].mean().reset_index()
    fig3 = px.box(feature_importance, x='Cluster', y=numerical_cols)
    st.plotly_chart(fig3)

    # Metric relationships
    st.subheader("How Customer Metrics Relate")
    fig4 = px.scatter_matrix(df, dimensions=numerical_cols[:5], color='Cluster')
    st.plotly_chart(fig4)

    # 3D visualization
    st.subheader("Customer Groups in 3D")
    pca_3d = PCA(n_components=3)
    df_pca_3d = pca_3d.fit_transform(df.drop(columns=['Cluster']))
    df_pca_3d = pd.DataFrame(df_pca_3d, columns=['PC1', 'PC2', 'PC3'])
    df_pca_3d['Cluster'] = df['Cluster']
    fig5 = px.scatter_3d(df_pca_3d, x='PC1', y='PC2', z='PC3', color='Cluster')
    st.plotly_chart(fig5)

    # Radar charts per group
    st.subheader("Strengths of Each Group")
    cluster_means = df.groupby('Cluster')[numerical_cols].mean().reset_index()
    radar_features = numerical_cols[:5]
    for cluster in cluster_means['Cluster']:
        fig6 = go.Figure()
        fig6.add_trace(go.Scatterpolar(
            r=cluster_means[cluster_means['Cluster'] == cluster][radar_features].values.flatten(),
            theta=[col.replace("_", " ").title() for col in radar_features],
            fill='toself',
            name=f'Group {cluster}'
        ))
        fig6.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title=f"Group {cluster} Profile"
        )
        st.plotly_chart(fig6)

    # Heatmap comparison
    st.subheader("Compare Groups Across All Metrics")
    fig7 = px.imshow(df.groupby('Cluster')[numerical_cols].mean(),
                     labels=dict(x="Metrics", y="Cluster", color="Value"),
                     title="Metric Comparison Across Groups")
    st.plotly_chart(fig7)

else:
    st.info("Please upload a CSV file to begin.")
