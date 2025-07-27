import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Unsupervised Learning Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .algorithm-header {
        font-size: 2rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .step-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def generate_sample_data(data_type, n_samples=300):
    """Generate different types of sample data"""
    if data_type == "Blobs":
        return make_blobs(n_samples=n_samples, centers=3, n_features=2, random_state=42, cluster_std=1.5)
    elif data_type == "Circles":
        return make_circles(n_samples=n_samples, factor=0.5, noise=0.1, random_state=42)
    elif data_type == "Moons":
        return make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    else:  # Random
        np.random.seed(42)
        return np.random.randn(n_samples, 2), np.zeros(n_samples)

def plot_elbow_method(X, max_k=10):
    """Plot elbow method for K-means"""
    inertias = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(k_range),
        y=inertias,
        mode='lines+markers',
        name='Inertia',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Elbow Method for Optimal K',
        xaxis_title='Number of Clusters (K)',
        yaxis_title='Inertia (Within-cluster Sum of Squares)',
        template='plotly_white'
    )
    
    return fig

def plot_silhouette_analysis(X, max_k=10):
    """Plot silhouette analysis"""
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(k_range),
        y=silhouette_scores,
        mode='lines+markers',
        name='Silhouette Score',
        line=dict(color='green', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Silhouette Analysis for Optimal K',
        xaxis_title='Number of Clusters (K)',
        yaxis_title='Silhouette Score',
        template='plotly_white'
    )
    
    return fig

def plot_dendrogram(X, linkage_method='ward'):
    """Plot dendrogram for hierarchical clustering"""
    linkage_matrix = linkage(X, method=linkage_method)
    
    fig = go.Figure()
    
    # Create dendrogram
    dn = dendrogram(linkage_matrix, no_plot=True)
    
    # Plot dendrogram
    for i, d in zip(dn['icoord'], dn['dcoord']):
        fig.add_trace(go.Scatter(
            x=i, y=d,
            mode='lines',
            line=dict(color='blue', width=2),
            showlegend=False
        ))
    
    fig.update_layout(
        title=f'Dendrogram (Linkage: {linkage_method})',
        xaxis_title='Sample Index',
        yaxis_title='Distance',
        template='plotly_white'
    )
    
    return fig

def kmeans_page():
    """K-Means Clustering Page"""
    st.markdown('<h1 class="algorithm-header">🎯 K-Means Clustering</h1>', unsafe_allow_html=True)
    
    # Algorithm Steps
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.markdown("""
    **📚 How K-Means Works:**
    1. **Choose K** - Decide how many clusters you want
    2. **Initialize** - Place K random points as initial cluster centers
    3. **Assign** - Put each data point into the closest cluster
    4. **Update** - Move cluster centers to the middle of their assigned points
    5. **Repeat** - Keep doing steps 3-4 until centers stop moving
    6. **Done** - You have your final clusters!
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔧 Settings")
        
        # Data selection
        data_type = st.selectbox("Choose Data Type:", ["Blobs", "Circles", "Moons", "Random"])
        n_samples = st.slider("Number of Data Points:", 1, 500, 300)
        
        # Generate data
        X, _ = generate_sample_data(data_type, n_samples)
        
        # Upload custom data option
        uploaded_file = st.file_uploader("Or Upload Your Own Data (CSV)", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) >= 2:
                selected_columns = st.multiselect("Select 2 Columns for Clustering:", numeric_cols, default=numeric_cols[:2])
                
                if len(selected_columns) == 2:
                    X = df[selected_columns].dropna().values
                    st.success(f"✅ Using columns: {selected_columns[0]} and {selected_columns[1]}")
                else:
                    st.error("❌ Please select exactly 2 numeric columns.")
            else:
                st.error("❌ Upload must contain at least 2 numeric columns.")
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K selection
        st.subheader("🎯 Choose Number of Clusters (K)")
        k_clusters = st.slider("Number of Clusters (K):", 2, 10, 3)
        
        # Show analysis methods
        if st.button("📊 Find Optimal K"):
            st.subheader("🔍 Elbow Method")
            elbow_fig = plot_elbow_method(X_scaled)
            st.plotly_chart(elbow_fig, use_container_width=True)
            
            st.subheader("🔍 Silhouette Analysis")
            silhouette_fig = plot_silhouette_analysis(X_scaled)
            st.plotly_chart(silhouette_fig, use_container_width=True)
    
    with col2:
        st.subheader("🎨 Visualization")
        
        # Perform K-means
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        centers = kmeans.cluster_centers_
        
        # Create visualization
        fig = px.scatter(
            x=X_scaled[:, 0], y=X_scaled[:, 1],
            color=labels,
            title=f'K-Means Clustering (K={k_clusters})',
            labels={'x': 'Feature 1', 'y': 'Feature 2'},
            color_continuous_scale='viridis'
        )
        
        # Add cluster centers
        fig.add_trace(go.Scatter(
            x=centers[:, 0], y=centers[:, 1],
            mode='markers',
            marker=dict(size=15, color='red', symbol='x'),
            name='Cluster Centers'
        ))
        
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show metrics
        silhouette_avg = silhouette_score(X_scaled, labels)
        inertia = kmeans.inertia_
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Silhouette Score</h3>
                <h2 style="color: green;">{silhouette_avg:.3f}</h2>
                <p>Higher is better (max: 1.0)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Inertia</h3>
                <h2 style="color: blue;">{inertia:.2f}</h2>
                <p>Lower is better</p>
            </div>
            """, unsafe_allow_html=True)

def hierarchical_page():
    """Hierarchical Clustering Page"""
    st.markdown('<h1 class="algorithm-header">🌳 Hierarchical Clustering</h1>', unsafe_allow_html=True)
    
    # Algorithm Steps
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.markdown("""
    **📚 How Hierarchical Clustering Works:**
    1. **Start** - Each data point is its own cluster
    2. **Find Closest** - Look for the two closest clusters
    3. **Merge** - Combine the two closest clusters into one
    4. **Repeat** - Keep merging until you have the desired number of clusters
    5. **Visualize** - Use a dendrogram to see the cluster hierarchy
    6. **Cut** - Choose where to "cut" the tree to get your final clusters
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔧 Settings")
        
        # Data selection
        data_type = st.selectbox("Choose Data Type:", ["Blobs", "Circles", "Moons", "Random"])
        n_samples = st.slider("Number of Data Points:", 1, 500, 300)
        
        # Generate data
        X, _ = generate_sample_data(data_type, n_samples)
        
        # Upload custom data option
        uploaded_file = st.file_uploader("Or Upload Your Own Data (CSV)", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 2:
                selected_columns = st.multiselect("Select 2 Columns for Clustering:", numeric_cols, default=numeric_cols[:2])
                
                if len(selected_columns) == 2:
                    X = df[selected_columns].dropna().values
                    st.success(f"✅ Using columns: {selected_columns[0]} and {selected_columns[1]}")
                else:
                    st.error("❌ Please select exactly 2 numeric columns.")
            else:
                st.error("❌ Upload must contain at least 2 numeric columns.")
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Clustering parameters
        st.subheader("🎯 Clustering Parameters")
        n_clusters = st.slider("Number of Clusters:", 2, 10, 3)
        linkage_method = st.selectbox("Linkage Method:", ["ward", "complete", "average", "single"])
        
        # Show dendrogram
        if st.button("🌳 Show Dendrogram"):
            st.subheader("🌳 Dendrogram")
            dendro_fig = plot_dendrogram(X_scaled, linkage_method)
            st.plotly_chart(dendro_fig, use_container_width=True)
    
    with col2:
        st.subheader("🎨 Visualization")
        
        # Perform hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = hierarchical.fit_predict(X_scaled)
        
        # Create visualization
        fig = px.scatter(
            x=X_scaled[:, 0], y=X_scaled[:, 1],
            color=labels,
            title=f'Hierarchical Clustering (Clusters={n_clusters}, Linkage={linkage_method})',
            labels={'x': 'Feature 1', 'y': 'Feature 2'},
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show metrics
        silhouette_avg = silhouette_score(X_scaled, labels)
        
        # Silhouette analysis
        st.subheader("🔍 Silhouette Analysis")
        silhouette_fig = plot_silhouette_analysis(X_scaled, max_k=10)
        st.plotly_chart(silhouette_fig, use_container_width=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Silhouette Score</h3>
            <h2 style="color: green;">{silhouette_avg:.3f}</h2>
            <p>Higher is better (max: 1.0)</p>
        </div>
        """, unsafe_allow_html=True)

def dbscan_page():
    """DBSCAN Clustering Page"""
    st.markdown('<h1 class="algorithm-header">🔍 DBSCAN Clustering</h1>', unsafe_allow_html=True)
    
    # Algorithm Steps
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.markdown("""
    **📚 How DBSCAN Works:**
    1. **Pick a Point** - Start with any unvisited data point
    2. **Check Neighbors** - Find all points within epsilon (ε) distance
    3. **Core Point?** - If it has enough neighbors (min_samples), it's a core point
    4. **Grow Cluster** - Add all reachable points to the same cluster
    5. **Border Points** - Points near core points but not core themselves
    6. **Noise Points** - Points that don't belong to any cluster (outliers)
    7. **Repeat** - Continue until all points are processed
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔧 Settings")
        
        # Data selection
        data_type = st.selectbox("Choose Data Type:", ["Blobs", "Circles", "Moons", "Random"])
        n_samples = st.slider("Number of Data Points:", 1, 500, 300)
        
        # Generate data
        X, _ = generate_sample_data(data_type, n_samples)
        
        # Upload custom data option
        uploaded_file = st.file_uploader("Or Upload Your Own Data (CSV)", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) >= 2:
                selected_columns = st.multiselect("Select 2 Columns for Clustering:", numeric_cols, default=numeric_cols[:2])
                
                if len(selected_columns) == 2:
                    X = df[selected_columns].dropna().values
                    st.success(f"✅ Using columns: {selected_columns[0]} and {selected_columns[1]}")
                else:
                    st.error("❌ Please select exactly 2 numeric columns.")
            else:
                st.error("❌ Upload must contain at least 2 numeric columns.")
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # DBSCAN parameters
        st.subheader("🎯 DBSCAN Parameters")
        st.markdown("**Epsilon (ε):** Maximum distance between two points to be neighbors")
        eps = st.slider("Epsilon (ε):", 0.1, 2.0, 0.5, 0.1)
        
        st.markdown("**Min Samples:** Minimum points needed to form a core point")
        min_samples = st.slider("Min Samples:", 2, 20, 5)
        
        # Help text
        st.info("""
        💡 **Tips:**
        - **Small ε**: More clusters, more noise
        - **Large ε**: Fewer clusters, less noise
        - **Small min_samples**: More core points
        - **Large min_samples**: Fewer core points
        """)
    
    with col2:
        st.subheader("🎨 Visualization")
        
        # Perform DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        # Count clusters and noise
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Create visualization
        fig = px.scatter(
            x=X_scaled[:, 0], y=X_scaled[:, 1],
            color=[f'Cluster {i}' if i != -1 else 'Noise' for i in labels],
            title=f'DBSCAN Clustering (ε={eps}, min_samples={min_samples})',
            labels={'x': 'Feature 1', 'y': 'Feature 2'}
        )
        
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Clusters Found</h3>
                <h2 style="color: blue;">{n_clusters}</h2>
                <p>Discovered clusters</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Noise Points</h3>
                <h2 style="color: red;">{n_noise}</h2>
                <p>Outliers detected</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m3:
            if n_clusters > 1:
                silhouette_avg = silhouette_score(X_scaled, labels)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Silhouette Score</h3>
                    <h2 style="color: green;">{silhouette_avg:.3f}</h2>
                    <p>Clustering quality</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Silhouette Score</h3>
                    <h2 style="color: gray;">N/A</h2>
                    <p>Need 2+ clusters</p>
                </div>
                """, unsafe_allow_html=True)

def main():
    """Main function"""
    st.markdown('<h1 class="main-header">🔍 Unsupervised Learning Visual Classifier</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    # 🔖 Personal Info Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👩‍💻 Developed by: **Stuti Agrawal**")
    st.sidebar.markdown("📧 Email: [stutiagrawal61@gmail.com](mailto:stutiagrawal61@gmail.com)")
    st.sidebar.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/stuti-agrawal-48918b27b/)")
    st.sidebar.markdown("[✍️ Medium](https://medium.com/@stutiagrawal61)")
    st.sidebar.markdown("[💻 GitHub](https://github.com/Stutii25)")
    st.sidebar.markdown("---")

    page = st.sidebar.selectbox(
        "Choose an Algorithm:",
        ["Home", "K-Means Clustering", "Hierarchical Clustering", "DBSCAN Clustering"]
    )
    
    if page == "Home":
        st.markdown("""
        ## Welcome to the Unsupervised Learning Visual Classifier! 🎓
        
        This interactive tool helps you understand three important clustering algorithms:
        
        ### 🎯 K-Means Clustering
        - Groups data into K clusters
        - Uses cluster centers (centroids)
        - Best for spherical clusters
        - **Tools:** Elbow Method, Silhouette Analysis
        
        ### 🌳 Hierarchical Clustering
        - Creates a tree of clusters
        - No need to specify K beforehand
        - Shows cluster relationships
        - **Tools:** Dendrogram, Silhouette Analysis
        
        ### 🔍 DBSCAN Clustering
        - Finds clusters of any shape
        - Automatically detects outliers
        - No need to specify number of clusters
        - **Tools:** Noise detection, Density-based grouping
        
        ---
        
        ### 🚀 How to Use:
        1. **Select an algorithm** from the sidebar
        2. **Choose your data** (sample data or upload your own)
        3. **Adjust parameters** to see how they affect clustering
        4. **Explore visualizations** to understand the results
        5. **Use analysis tools** to find optimal parameters
        
        ### 📊 What You'll Learn:
        - How each algorithm works step-by-step
        - When to use each algorithm
        - How to choose optimal parameters
        - How to interpret clustering results
        - How to evaluate clustering quality
        
        **👆 Choose an algorithm from the sidebar to get started!**
        """)
        
        # Add some sample visualizations
        st.subheader("🎨 Sample Data Types Available:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            X_blobs, _ = make_blobs(n_samples=150, centers=3, random_state=42)
            fig1 = px.scatter(x=X_blobs[:, 0], y=X_blobs[:, 1], title="Blobs Data")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            X_circles, _ = make_circles(n_samples=150, factor=0.5, noise=0.1, random_state=42)
            fig2 = px.scatter(x=X_circles[:, 0], y=X_circles[:, 1], title="Circles Data")
            st.plotly_chart(fig2, use_container_width=True)
        
        with col3:
            X_moons, _ = make_moons(n_samples=150, noise=0.1, random_state=42)
            fig3 = px.scatter(x=X_moons[:, 0], y=X_moons[:, 1], title="Moons Data")
            st.plotly_chart(fig3, use_container_width=True)
    
    elif page == "K-Means Clustering":
        kmeans_page()
    
    elif page == "Hierarchical Clustering":
        hierarchical_page()
    
    elif page == "DBSCAN Clustering":
        dbscan_page()

def unsupervised_main():
    main()
