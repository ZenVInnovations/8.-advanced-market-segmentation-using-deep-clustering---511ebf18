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

# Suppress TensorFlow warnings
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
    with st.spinner('Training autoencoder model (this may take a few moments)...'):
        split_idx = int(0.8 * len(df_pca))
        X_train, X_val = df_pca[:split_idx], df_pca[split_idx:]
        autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, validation_data=(X_val, X_val), verbose=0)
    
    st.success("Autoencoder training complete!")

    # Get encoded representation
    encoder = Model(input_layer, encoded)
    X_encoded = encoder.predict(df_pca)

    # Choose number of clusters
    n_clusters = st.number_input("Select number of clusters", min_value=2, max_value=10, value=5)

    # K-Means clustering
    with st.spinner('Performing clustering...'):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_encoded)
    
    # Save models
    try:
        autoencoder.save("autoencoder_model.keras")
        with open("kmeans_model.pkl", "wb") as f:
            pickle.dump(kmeans, f)
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open("pca.pkl", "wb") as f:
            pickle.dump(pca, f)
        st.success("Models saved successfully!")
    except Exception as e:
        st.warning(f"Could not save models: {e}")

    # Assign clusters
    df['Cluster'] = kmeans.labels_

    st.header("📈 Segmentation Results")

    # Group distribution
    st.subheader("How Many Customers Are in Each Group?")
    cluster_counts = df['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Customer Group', 'Number of Customers']
    fig1 = px.bar(cluster_counts, x='Customer Group', y='Number of Customers', color='Customer Group')
    st.plotly_chart(fig1, use_container_width=True)

    # 2D visualization
    st.subheader("Customer Groups in 2D Space")
    pca_2d = PCA(n_components=2)
    df_pca_2d = pca_2d.fit_transform(df.drop(columns=['Cluster']))
    df_pca_2d = pd.DataFrame(df_pca_2d, columns=['PC1', 'PC2'])
    df_pca_2d['Cluster'] = df['Cluster']
    fig2 = px.scatter(df_pca_2d, x='PC1', y='PC2', color='Cluster', title="2D Projection of Clusters")
    st.plotly_chart(fig2, use_container_width=True)

    # Key metrics across groups
    st.subheader("What Makes Each Group Different?")
    # Use up to 8 numerical columns to avoid overcrowding
    display_cols = numerical_cols[:8] if len(numerical_cols) > 8 else numerical_cols
    feature_importance = df.groupby('Cluster')[display_cols].mean().reset_index()
    fig3 = px.box(df, x='Cluster', y=display_cols)
    st.plotly_chart(fig3, use_container_width=True)

    # Metric relationships
    st.subheader("How Customer Metrics Relate")
    # Use up to 5 numerical columns for scatter matrix
    scatter_cols = numerical_cols[:5] if len(numerical_cols) > 5 else numerical_cols
    fig4 = px.scatter_matrix(df, dimensions=scatter_cols, color='Cluster')
    st.plotly_chart(fig4, use_container_width=True)

    # 3D visualization
    st.subheader("Customer Groups in 3D")
    pca_3d = PCA(n_components=3)
    df_pca_3d = pca_3d.fit_transform(df.drop(columns=['Cluster']))
    df_pca_3d = pd.DataFrame(df_pca_3d, columns=['PC1', 'PC2', 'PC3'])
    df_pca_3d['Cluster'] = df['Cluster']
    fig5 = px.scatter_3d(df_pca_3d, x='PC1', y='PC2', z='PC3', color='Cluster')
    st.plotly_chart(fig5, use_container_width=True)

    # Radar charts per group
    st.subheader("Strengths of Each Group")
    cluster_means = df.groupby('Cluster')[numerical_cols].mean().reset_index()
    # Choose up to 8 features for radar chart to keep it readable
    radar_features = numerical_cols[:8] if len(numerical_cols) > 8 else numerical_cols
    
    radar_cols = st.columns(min(3, n_clusters))
    for i, cluster in enumerate(sorted(cluster_means['Cluster'].unique())):
        col_idx = i % len(radar_cols)
        with radar_cols[col_idx]:
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
            st.plotly_chart(fig6, use_container_width=True)

    # Heatmap comparison
    st.subheader("Compare Groups Across All Metrics")
    fig7 = px.imshow(df.groupby('Cluster')[numerical_cols].mean(),
                     labels=dict(x="Metrics", y="Cluster", color="Value"),
                     title="Metric Comparison Across Groups")
    st.plotly_chart(fig7, use_container_width=True)
    
    # Export results
    if st.button("Export Segmentation Results"):
        try:
            # Add cluster labels to original data
            result_df = df.copy()
            result_df.to_csv("customer_segments_results.csv", index=False)
            st.success("Results exported to 'customer_segments_results.csv'")
        except Exception as e:
            st.error(f"Failed to export results: {e}")

else:
    st.info("Please upload a CSV file to begin.")
    
    # Show example application
    st.subheader("How It Works")
    st.markdown("""
    1. *Upload Data*: Start by uploading a CSV file containing your customer data
    2. *Data Processing*: The app automatically handles missing values and encodes categorical features
    3. *Dimensionality Reduction*: Uses PCA to reduce complexity while preserving information
    4. *Deep Feature Extraction*: An autoencoder learns meaningful patterns in your data
    5. *Clustering*: Customers are grouped based on their characteristics
    6. *Visualization*: Explore the segments through interactive charts
    
    This tool combines deep learning and clustering to find natural customer segments in your data!
    """)