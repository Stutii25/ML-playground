import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_curve, auc, 
                           mean_squared_error, mean_absolute_error, r2_score)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification, make_regression, make_circles, make_moons
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
def main():
    st.set_page_config(
        page_title="Supervised Learning Visual Classifier",
        page_icon="🎯",
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
        .classification-card {
            background-color: #e8f5e8;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #28a745;
            margin: 1rem 0;
        }
        .regression-card {
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    def generate_sample_data(data_type, n_samples=300, noise=0.1, problem_type="classification"):
        """Generate different types of sample data for supervised learning"""
        np.random.seed(42)
        
        if problem_type == "classification":
            if data_type == "Linear Separable":
                X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                                        n_informative=2, n_classes=2, n_clusters_per_class=1, 
                                        random_state=42)
            elif data_type == "Circles":
                X, y = make_circles(n_samples=n_samples, factor=0.5, noise=noise, random_state=42)
            elif data_type == "Moons":
                X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
            elif data_type == "Multi-class":
                X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                                        n_informative=2, n_classes=3, n_clusters_per_class=1, 
                                        random_state=42)
            else:  # Random
                X = np.random.randn(n_samples, 2)
                y = np.random.randint(0, 2, n_samples)
        else:  # regression
            if data_type == "Linear":
                X, y = make_regression(n_samples=n_samples, n_features=2, noise=noise*10, 
                                    random_state=42)
            elif data_type == "Polynomial":
                X = np.random.randn(n_samples, 2)
                y = X[:, 0]**2 + X[:, 1]**2 + np.random.randn(n_samples) * noise
            elif data_type == "Sine Wave":
                X = np.random.randn(n_samples, 2)
                y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.randn(n_samples) * noise
            else:  # Random
                X = np.random.randn(n_samples, 2)
                y = np.random.randn(n_samples)
        
        return X, y

    def plot_classification_boundary(X, y, model, title="Decision Boundary"):
        """Plot decision boundary for classification"""
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig = go.Figure()
        
        # Add contour for decision boundary
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=Z,
            showscale=False,
            opacity=0.3,
            colorscale='RdYlBu'
        ))
        
        # Add scatter plot for data points
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i in range(len(np.unique(y))):
            mask = y == i
            fig.add_trace(go.Scatter(
                x=X[mask, 0], y=X[mask, 1],
                mode='markers',
                name=f'Class {i}',
                marker=dict(color=colors[i % len(colors)], size=8)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            template='plotly_white'
        )
        
        return fig

    def plot_regression_line(X, y, model, title="Regression Line"):
        """Plot regression line (for 2D data, we'll use first feature)"""
        if X.shape[1] > 1:
            # Use first feature for visualization
            X_plot = X[:, 0].reshape(-1, 1)
            y_pred = model.predict(X)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=X_plot.flatten(), y=y,
                mode='markers',
                name='Actual',
                marker=dict(color='blue', size=8)
            ))
            fig.add_trace(go.Scatter(
                x=X_plot.flatten(), y=y_pred,
                mode='markers',
                name='Predicted',
                marker=dict(color='red', size=8)
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Feature 1',
                yaxis_title='Target',
                template='plotly_white'
            )
        else:
            X_plot = X
            y_pred = model.predict(X)
            
            # Sort for line plot
            sort_idx = np.argsort(X_plot.flatten())
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=X_plot[sort_idx].flatten(), y=y[sort_idx],
                mode='markers',
                name='Actual',
                marker=dict(color='blue', size=8)
            ))
            fig.add_trace(go.Scatter(
                x=X_plot[sort_idx].flatten(), y=y_pred[sort_idx],
                mode='lines',
                name='Predicted',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Feature',
                yaxis_title='Target',
                template='plotly_white'
            )
        
        return fig

    def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f'Predicted {i}' for i in range(len(cm))],
            y=[f'Actual {i}' for i in range(len(cm))],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 20}
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Predicted',
            yaxis_title='Actual',
            template='plotly_white'
        )
        
        return fig

    def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve"):
        """Plot ROC curve for binary classification"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.2f})',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            template='plotly_white'
        )
        
        return fig

    def plot_learning_curve(model, X, y, title="Learning Curve"):
        """Plot learning curve"""
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=np.mean(train_scores, axis=1),
            mode='lines+markers',
            name='Training Score',
            line=dict(color='blue', width=2),
            error_y=dict(array=np.std(train_scores, axis=1))
        ))
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=np.mean(val_scores, axis=1),
            mode='lines+markers',
            name='Validation Score',
            line=dict(color='red', width=2),
            error_y=dict(array=np.std(val_scores, axis=1))
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Training Set Size',
            yaxis_title='Accuracy Score',
            template='plotly_white'
        )
        
        return fig

    def logistic_regression_page():
        """Logistic Regression Page"""
        st.markdown('<h1 class="algorithm-header">📊 Logistic Regression</h1>', unsafe_allow_html=True)
        
        # Algorithm Steps
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **📚 How Logistic Regression Works:**
        1. **Linear Combination** - Multiply each feature by a weight and add them up
        2. **Sigmoid Function** - Apply sigmoid function to get probability between 0 and 1
        3. **Decision Boundary** - If probability > 0.5, predict class 1, else class 0
        4. **Cost Function** - Use log-likelihood to measure how wrong predictions are
        5. **Gradient Descent** - Adjust weights to minimize the cost function
        6. **Repeat** - Keep adjusting weights until they don't change much
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🔧 Settings")
            
            # Data selection
            data_type = st.selectbox("Choose Data Type:", ["Linear Separable", "Circles", "Moons", "Multi-class"])
            n_samples = st.slider("Number of Data Points:", 1, 1000, 300)
            noise = st.slider("Noise Level:", 0.0, 0.5, 0.1)
            
            # Generate data
            X, y = generate_sample_data(data_type, n_samples, noise, "classification")
            
            # Upload custom data option
            uploaded_file = st.file_uploader("📂 Upload Your Own Data (CSV)", type=['csv'])
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    st.success("✅ Data loaded successfully!")
                    
                    feature_cols = st.multiselect("📊 Select Feature Columns:", numeric_cols, default=numeric_cols[:-1])
                    target_col = st.selectbox("🎯 Select Target Column:", numeric_cols, index=len(numeric_cols) - 1)
                    
                    if target_col in feature_cols:
                        st.error("❌ Target column must be different from feature columns.")
                    elif len(feature_cols) == 0:
                        st.error("❌ Please select at least one feature column.")
                    else:
                        X = df[feature_cols].dropna().values
                        y = df[target_col].dropna().values
                        
                        # Encode target if it's categorical
                        if y.dtype == 'object':
                            le = LabelEncoder()
                            y = le.fit_transform(y)
                        # ✅ Add this check for classification compatibility
                        from sklearn.utils.multiclass import type_of_target
                        target_type = type_of_target(y)
                        if target_type not in ['binary', 'multiclass']:
                            st.error("❌ Logistic Regression requires a **categorical target** (e.g., class labels like 0, 1, 2). Your selected target seems continuous.")
                            st.stop()
                        st.success(f"✅ Using features: {', '.join(feature_cols)} and target: {target_col}")
                else:
                    st.error("❌ Data must have at least 2 numeric columns.")

            # Model parameters
            st.subheader("🎯 Model Parameters")
            C = st.slider("Regularization (C):", 0.01, 10.0, 1.0)
            max_iter = st.slider("Max Iterations:", 1, 1000, 100)
            
            # Train-test split
            test_size = st.slider("Test Size:", 0.1, 0.5, 0.2)
            
            st.info("""
            💡 **Parameter Tips:**
            - **Higher C**: Less regularization, more complex model
            - **Lower C**: More regularization, simpler model
            - **More iterations**: Better convergence
            """)
        
        with col2:
            st.subheader("🎨 Visualization & Results")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Decision boundary plot (only for 2D data)
            if X.shape[1] == 2:
                fig = plot_classification_boundary(X_test_scaled, y_test, model, 
                                                f"Logistic Regression Decision Boundary (C={C})")
                st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Display metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Accuracy</h3>
                    <h2 style="color: green;">{accuracy:.3f}</h2>
                    <p>Correct predictions</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Precision</h3>
                    <h2 style="color: blue;">{precision:.3f}</h2>
                    <p>True positive rate</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Recall</h3>
                    <h2 style="color: orange;">{recall:.3f}</h2>
                    <p>Sensitivity</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>F1-Score</h3>
                    <h2 style="color: purple;">{f1:.3f}</h2>
                    <p>Harmonic mean</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional visualizations
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                # Confusion Matrix
                cm_fig = plot_confusion_matrix(y_test, y_pred, "Confusion Matrix")
                st.plotly_chart(cm_fig, use_container_width=True)
            
            with col_v2:
                # ROC Curve (only for binary classification)
                if len(np.unique(y)) == 2:
                    roc_fig = plot_roc_curve(y_test, y_pred_proba[:, 1], "ROC Curve")
                    st.plotly_chart(roc_fig, use_container_width=True)
                else:
                    st.info("ROC Curve is only available for binary classification")
            
            # Learning Curve
            if st.button("📈 Show Learning Curve"):
                lc_fig = plot_learning_curve(model, X_train_scaled, y_train, "Learning Curve")
                st.plotly_chart(lc_fig, use_container_width=True)

    def linear_regression_page():
        """Linear Regression Page"""
        st.markdown('<h1 class="algorithm-header">📈 Linear Regression</h1>', unsafe_allow_html=True)
        
        # Algorithm Steps
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **📚 How Linear Regression Works:**
        1. **Linear Equation** - Find the best line: y = mx + b (or y = w₁x₁ + w₂x₂ + ... + b)
        2. **Cost Function** - Measure how far predictions are from actual values (Mean Squared Error)
        3. **Gradient Descent** - Adjust weights to minimize the cost function
        4. **Update Weights** - Move in the direction that reduces error
        5. **Repeat** - Keep adjusting until weights don't change much
        6. **Final Model** - Use the best weights to make predictions
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🔧 Settings")
            
            # Data selection
            data_type = st.selectbox("Choose Data Type:", ["Linear", "Polynomial", "Sine Wave", "Random"])
            n_samples = st.slider("Number of Data Points:", 1, 1000, 300)
            noise = st.slider("Noise Level:", 0.0, 2.0, 0.1)
            
            # Generate data
            X, y = generate_sample_data(data_type, n_samples, noise, "regression")
            
            # Upload custom data option
            uploaded_file = st.file_uploader("📂 Upload Your Own Data (CSV)", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    st.success("✅ Data loaded successfully!")
                    feature_cols = st.multiselect("📊 Select Feature Columns:", numeric_cols, default=numeric_cols[:-1])
                    target_col = st.selectbox("🎯 Select Target Column:", numeric_cols, index=len(numeric_cols)-1)
                    
                    if target_col in feature_cols:
                        st.error("❌ Target column must be different from feature columns.")
                    elif len(feature_cols) == 0:
                        st.error("❌ Please select at least one feature column.")
                    else:
                        X = df[feature_cols].dropna().values
                        y = df[target_col].dropna().values
                        st.success(f"✅ Using features: {', '.join(feature_cols)} and target: {target_col}")
                else:
                    st.error("❌ Data must have at least 2 numeric columns.")
            
            # Model parameters
            st.subheader("🎯 Model Parameters")
            fit_intercept = st.checkbox("Fit Intercept", value=True)
            
            # Train-test split
            test_size = st.slider("Test Size:", 0.1, 0.5, 0.2)
            
            st.info("""
            💡 **Tips:**
            - **Fit Intercept**: Whether to calculate intercept (bias term)
            - **Linear data**: Works best with linear relationships
            - **Polynomial data**: May need feature engineering
            """)
        
        with col2:
            st.subheader("🎨 Visualization & Results")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = LinearRegression(fit_intercept=fit_intercept)
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Regression plot
            fig = plot_regression_line(X_test_scaled, y_test, model, "Linear Regression")
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Display metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>R² Score</h3>
                    <h2 style="color: green;">{r2:.3f}</h2>
                    <p>Explained variance</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>MSE</h3>
                    <h2 style="color: red;">{mse:.3f}</h2>
                    <p>Mean Squared Error</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>MAE</h3>
                    <h2 style="color: orange;">{mae:.3f}</h2>
                    <p>Mean Absolute Error</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>RMSE</h3>
                    <h2 style="color: blue;">{rmse:.3f}</h2>
                    <p>Root Mean Squared Error</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Residual plot
            residuals = y_test - y_pred
            fig_residuals = go.Figure()
            fig_residuals.add_trace(go.Scatter(
                x=y_pred, y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='blue', size=8)
            ))
            fig_residuals.add_trace(go.Scatter(
                x=[y_pred.min(), y_pred.max()], y=[0, 0],
                mode='lines',
                name='Zero Line',
                line=dict(color='red', width=2, dash='dash')
            ))
            fig_residuals.update_layout(
                title='Residual Plot',
                xaxis_title='Predicted Values',
                yaxis_title='Residuals',
                template='plotly_white'
            )
            st.plotly_chart(fig_residuals, use_container_width=True)
            
            # Learning Curve
            if st.button("📈 Show Learning Curve"):
                lc_fig = plot_learning_curve(model, X_train_scaled, y_train, "Learning Curve")
                st.plotly_chart(lc_fig, use_container_width=True)

    def decision_tree_page():
        """Decision Tree Page"""
        st.markdown('<h1 class="algorithm-header">🌳 Decision Tree</h1>', unsafe_allow_html=True)
        
        # Algorithm Steps
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **📚 How Decision Tree Works:**
        1. **Choose Best Split** - Find the feature and value that best separates the data
        2. **Split Data** - Divide data into two groups based on the chosen feature
        3. **Measure Purity** - Use metrics like Gini impurity or entropy to measure how "pure" each group is
        4. **Repeat Process** - Apply the same process to each new group (recursive splitting)
        5. **Stop Splitting** - When groups are pure enough or meet stopping criteria
        6. **Make Predictions** - Follow the tree branches to make predictions for new data
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🔧 Settings")
            
            # Problem type
            problem_type = st.selectbox("Problem Type:", ["Classification", "Regression"])
            
            # Data selection
            if problem_type == "Classification":
                data_type = st.selectbox("Choose Data Type:", ["Linear Separable", "Circles", "Moons", "Multi-class"])
            else:
                data_type = st.selectbox("Choose Data Type:", ["Linear", "Polynomial", "Sine Wave", "Random"])
            
            n_samples = st.slider("Number of Data Points:", 1, 1000, 300)
            noise = st.slider("Noise Level:", 0.0, 0.5, 0.1)
            
            # Generate data
            X, y = generate_sample_data(data_type, n_samples, noise, problem_type.lower())
            
            # Upload custom data option
            uploaded_file = st.file_uploader("📂 Upload Your Own Data (CSV)", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    st.success("✅ Data loaded successfully!")
                    feature_cols = st.multiselect("📊 Select Feature Columns:", numeric_cols, default=numeric_cols[:-1])
                    target_col = st.selectbox("🎯 Select Target Column:", numeric_cols, index=len(numeric_cols)-1)
                    
                    if target_col in feature_cols:
                        st.error("❌ Target column must be different from feature columns.")
                    elif len(feature_cols) == 0:
                        st.error("❌ Please select at least one feature column.")
                    else:
                        X = df[feature_cols].dropna().values
                        y = df[target_col].dropna().values
                        st.success(f"✅ Using features: {', '.join(feature_cols)} and target: {target_col}")
                else:
                    st.error("❌ Data must have at least 2 numeric columns.")
            
            # Model parameters
            st.subheader("🎯 Model Parameters")
            max_depth = st.slider("Max Depth:", 1, 20, 5)
            min_samples_split = st.slider("Min Samples Split:", 2, 20, 2)
            min_samples_leaf = st.slider("Min Samples Leaf:", 1, 20, 1)
            
            if problem_type == "Classification":
                criterion = st.selectbox("Criterion:", ["gini", "entropy"])
            else:
                criterion = st.selectbox("Criterion:", ["squared_error", "absolute_error"])
            
            # Train-test split
            test_size = st.slider("Test Size:", 0.1, 0.5, 0.2)
            
            st.info("""
            💡 **Parameter Tips:**
            - **Max Depth**: Deeper trees can overfit
            - **Min Samples Split**: Higher values prevent overfitting
            - **Gini vs Entropy**: Both measure impurity, entropy is more computationally expensive
            """)
        
        with col2:
            st.subheader("🎨 Visualization & Results")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Train model
            if problem_type == "Classification":
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    criterion=criterion,
                    random_state=42
                )
            else:
                model = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    criterion=criterion,
                    random_state=42
                )
            
            model.fit(X_train, y_train)
            # Add this code right after the model training (around line 94, after model.fit(X_train, y_train))

            # Decision Tree Visualization
            st.subheader("🌳 Decision Tree Structure")

            # Show tree structure only for reasonable sized trees to avoid clutter
            if max_depth <= 6 and len(X_train) <= 500 and X.shape[1] <= 10:
                try:
                    # Create the tree plot
                    fig, ax = plt.subplots(figsize=(15, 10))
                    
                    # Generate feature names
                    if uploaded_file is not None and 'feature_cols' in locals():
                        feature_names = feature_cols
                    else:
                        feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
                    
                    # Generate class names for classification
                    if problem_type == "Classification":
                        class_names = [str(cls) for cls in np.unique(y_train)]
                    else:
                        class_names = None
                    
                    # Plot the decision tree
                    plot_tree(model,
                            filled=True,
                            rounded=True,
                            feature_names=feature_names,
                            class_names=class_names,
                            fontsize=8,
                            ax=ax)
                    
                    ax.set_title(f"Decision Tree Structure (Depth: {max_depth})", fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                except Exception as e:
                    st.error(f"Error creating tree visualization: {str(e)}")
                    st.info("Tree structure too complex to visualize. Check the text rules below.")
                
                # Add tree rules as text
                with st.expander("📋 Decision Tree Rules (Text Format)"):
                    tree_rules = export_text(model, 
                                            feature_names=feature_names,
                                            show_weights=True)
                    st.code(tree_rules, language='text')

            else:
                st.info(f"""
                🔍 **Tree visualization not shown because:**
                - Tree depth > 6 (current: {max_depth}) or
                - Too many samples > 500 (current: {len(X_train)}) or  
                - Too many features > 10 (current: {X.shape[1]})
                
                This prevents cluttered visualization. You can still see the tree rules below.
                """)
                
                # Show simplified tree info
                st.write(f"**Tree Statistics:**")
                st.write(f"- Tree depth: {model.get_depth()}")
                st.write(f"- Number of leaves: {model.get_n_leaves()}")
                
                # Show tree rules in text format
                with st.expander("📋 Decision Tree Rules (Text Format)"):
                    if uploaded_file is not None and 'feature_cols' in locals():
                        feature_names = feature_cols
                    else:
                        feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
                        
                    tree_rules = export_text(model, 
                                            feature_names=feature_names,
                                            show_weights=True,
                                            max_depth=5)  # Limit depth for readability
                    st.code(tree_rules, language='text')

            # Add tree complexity metrics
            st.subheader("📊 Tree Complexity Metrics")
            col_t1, col_t2, col_t3 = st.columns(3)

            with col_t1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Tree Depth</h3>
                    <h2 style="color: blue;">{model.get_depth()}</h2>
                    <p>Maximum depth reached</p>
                </div>
                """, unsafe_allow_html=True)

            with col_t2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Number of Leaves</h3>
                    <h2 style="color: green;">{model.get_n_leaves()}</h2>
                    <p>Final decision nodes</p>
                </div>
                """, unsafe_allow_html=True)

            with col_t3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Total Nodes</h3>
                    <h2 style="color: orange;">{model.tree_.node_count}</h2>
                    <p>All decision + leaf nodes</p>
                </div>
                """, unsafe_allow_html=True)
                        
            # Predictions
            y_pred = model.predict(X_test)
            
            # Visualization based on problem type
            if problem_type == "Classification":
                if X.shape[1] == 2:
                    fig = plot_classification_boundary(X_test, y_test, model, 
                                                    f"Decision Tree Classification (Max Depth={max_depth})")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Classification metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Display metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Accuracy</h3>
                        <h2 style="color: green;">{accuracy:.3f}</h2>
                        <p>Correct predictions</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Precision</h3>
                        <h2 style="color: blue;">{precision:.3f}</h2>
                        <p>True positive rate</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Recall</h3>
                        <h2 style="color: orange;">{recall:.3f}</h2>
                        <p>Sensitivity</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>F1-Score</h3>
                        <h2 style="color: purple;">{f1:.3f}</h2>
                        <p>Harmonic mean</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confusion Matrix
                cm_fig = plot_confusion_matrix(y_test, y_pred, "Confusion Matrix")
                st.plotly_chart(cm_fig, use_container_width=True)
            
            else:  # Regression
                fig = plot_regression_line(X_test, y_test, model, "Decision Tree Regression")
                st.plotly_chart(fig, use_container_width=True)
                
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Display metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>R² Score</h3>
                        <h2 style="color: green;">{r2:.3f}</h2>
                        <p>Explained variance</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>MSE</h3>
                        <h2 style="color: red;">{mse:.3f}</h2>
                        <p>Mean Squared Error</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>MAE</h3>
                        <h2 style="color: orange;">{mae:.3f}</h2>
                        <p>Mean Absolute Error</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>RMSE</h3>
                        <h2 style="color: blue;">{rmse:.3f}</h2>
                        <p>Root Mean Squared Error</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Feature importance
            if X.shape[1] <= 10:  # Only show for reasonable number of features
                feature_importance = model.feature_importances_
                feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
                
                fig_importance = go.Figure()
                fig_importance.add_trace(go.Bar(
                    x=feature_names,
                    y=feature_importance,
                    name='Feature Importance'
                ))
                fig_importance.update_layout(
                    title='Feature Importance',
                    xaxis_title='Features',
                    yaxis_title='Importance',
                    template='plotly_white'
                )
                st.plotly_chart(fig_importance, use_container_width=True)

    def random_forest_page():
        """Random Forest Page"""
        st.markdown('<h1 class="algorithm-header">🌲 Random Forest</h1>', unsafe_allow_html=True)
        
        # Algorithm Steps
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **📚 How Random Forest Works:**
        1. **Bootstrap Sampling** - Create multiple random samples from the original data
        2. **Random Feature Selection** - For each split, randomly select a subset of features
        3. **Build Trees** - Train a decision tree on each bootstrap sample
        4. **Combine Results** - Average predictions (regression) or vote (classification)
        5. **Reduce Overfitting** - Multiple trees reduce the chance of overfitting
        6. **Final Prediction** - Use the combined result from all trees
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🔧 Settings")
            
            # Problem type
            problem_type = st.selectbox("Problem Type:", ["Classification", "Regression"])
            
            # Data selection
            if problem_type == "Classification":
                data_type = st.selectbox("Choose Data Type:", ["Linear Separable", "Circles", "Moons", "Multi-class"])
            else:
                data_type = st.selectbox("Choose Data Type:", ["Linear", "Polynomial", "Sine Wave", "Random"])
            
            n_samples = st.slider("Number of Data Points:", 1, 1000, 300)
            noise = st.slider("Noise Level:", 0.0, 0.5, 0.1)
            
            # Generate data
            X, y = generate_sample_data(data_type, n_samples, noise, problem_type.lower())
            
            # Upload custom data option
            uploaded_file = st.file_uploader("📂 Upload Your Own Data (CSV)", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    st.success("✅ Data loaded successfully!")
                    feature_cols = st.multiselect("📊 Select Feature Columns:", numeric_cols, default=numeric_cols[:-1])
                    target_col = st.selectbox("🎯 Select Target Column:", numeric_cols, index=len(numeric_cols)-1)
                    
                    if target_col in feature_cols:
                        st.error("❌ Target column must be different from feature columns.")
                    elif len(feature_cols) == 0:
                        st.error("❌ Please select at least one feature column.")
                    else:
                        X = df[feature_cols].dropna().values
                        y = df[target_col].dropna().values
                        st.success(f"✅ Using features: {', '.join(feature_cols)} and target: {target_col}")
                else:
                    st.error("❌ Data must have at least 2 numeric columns.")
            
            # Model parameters
            st.subheader("🎯 Model Parameters")
            n_estimators = st.slider("Number of Trees:", 10, 200, 100)
            max_depth = st.slider("Max Depth:", 1, 20, 5)
            min_samples_split = st.slider("Min Samples Split:", 2, 20, 2)
            max_features = st.selectbox("Max Features:", ["sqrt", "log2", "auto"])
            
            # Train-test split
            test_size = st.slider("Test Size:", 0.1, 0.5, 0.2)
            
            st.info("""
            💡 **Parameter Tips:**
            - **More Trees**: Better performance but slower training
            - **Max Features**: Controls randomness in feature selection
            - **Max Depth**: Deeper trees can overfit
            """)
        
        with col2:
            st.subheader("🎨 Visualization & Results")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Train model
            if problem_type == "Classification":
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    max_features=max_features,
                    random_state=42
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    max_features=max_features,
                    random_state=42
                )
            
            model.fit(X_train, y_train)
                # Random Forest Specific Visualizations
            # Place this code after model.fit(X_train, y_train) and before predictions in your Random Forest page

            # ========================================
            # 1. INDIVIDUAL TREES VISUALIZATION
            # ========================================
            st.subheader("🌲 Individual Trees in the Forest")

            # Show first few trees if reasonable size
            n_trees_to_show = min(3, model.n_estimators)
            show_individual_trees = (max_depth <= 5 and len(X_train) <= 300 and X.shape[1] <= 8)

            if show_individual_trees:
                st.info(f"Showing first {n_trees_to_show} trees out of {model.n_estimators} total trees")
                
                for i in range(n_trees_to_show):
                    with st.expander(f"🌳 Tree #{i+1} Structure"):
                        try:
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            # Get individual tree
                            individual_tree = model.estimators_[i]
                            
                            # Prepare feature names
                            if uploaded_file is not None and 'feature_cols' in locals():
                                feature_names = feature_cols
                            else:
                                feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
                            
                            # Prepare class names for classification
                            if problem_type == "Classification":
                                class_names = [str(cls) for cls in np.unique(y_train)]
                            else:
                                class_names = None
                            
                            # Plot individual tree
                            plot_tree(
                                individual_tree,
                                filled=True,
                                rounded=True,
                                feature_names=feature_names,
                                class_names=class_names,
                                fontsize=6,
                                ax=ax
                            )
                            
                            ax.set_title(f"Decision Tree #{i+1} in Random Forest", 
                                    fontsize=14, fontweight='bold')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                            
                        except Exception as e:
                            st.error(f"Could not visualize tree #{i+1}: {str(e)}")
            else:
                st.info(f"""
                🔍 **Individual tree visualization not shown:**
                - Trees are too complex or dataset too large
                - Current: {model.n_estimators} trees, depth ≤{max_depth}, {len(X_train)} samples, {X.shape[1]} features
                - Recommended: ≤5 depth, ≤300 samples, ≤8 features for clear visualization
                """)

            # ========================================
            # 2. FOREST STATISTICS & STRUCTURE
            # ========================================
            st.subheader("🌳 Forest Structure & Statistics")

            col_f1, col_f2, col_f3, col_f4 = st.columns(4)

            with col_f1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Total Trees</h3>
                    <h2 style="color: green;">{model.n_estimators}</h2>
                    <p>Trees in forest</p>
                </div>
                """, unsafe_allow_html=True)

            with col_f2:
                # Calculate average tree depth
                avg_depth = np.mean([tree.get_depth() for tree in model.estimators_])
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Avg Tree Depth</h3>
                    <h2 style="color: blue;">{avg_depth:.1f}</h2>
                    <p>Average depth</p>
                </div>
                """, unsafe_allow_html=True)

            with col_f3:
                # Calculate average number of leaves
                avg_leaves = np.mean([tree.get_n_leaves() for tree in model.estimators_])
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Avg Leaves</h3>
                    <h2 style="color: orange;">{avg_leaves:.0f}</h2>
                    <p>Average leaf nodes</p>
                </div>
                """, unsafe_allow_html=True)

            with col_f4:
                # Calculate total nodes across all trees
                total_nodes = sum([tree.tree_.node_count for tree in model.estimators_])
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Total Nodes</h3>
                    <h2 style="color: purple;">{total_nodes:,}</h2>
                    <p>All nodes combined</p>
                </div>
                """, unsafe_allow_html=True)

            # ========================================
            # 3. TREE DIVERSITY ANALYSIS
            # ========================================
            st.subheader("📊 Forest Diversity Analysis")

            # Tree depth distribution
            tree_depths = [tree.get_depth() for tree in model.estimators_]
            tree_leaves = [tree.get_n_leaves() for tree in model.estimators_]

            col_div1, col_div2 = st.columns(2)

            with col_div1:
                # Tree depth distribution
                fig_depth = go.Figure()
                fig_depth.add_trace(go.Histogram(
                    x=tree_depths,
                    nbinsx=min(20, len(set(tree_depths))),
                    name='Tree Depths',
                    marker_color='lightblue'
                ))
                fig_depth.update_layout(
                    title='Distribution of Tree Depths',
                    xaxis_title='Tree Depth',
                    yaxis_title='Number of Trees',
                    template='plotly_white'
                )
                st.plotly_chart(fig_depth, use_container_width=True)

            with col_div2:
                # Tree leaves distribution
                fig_leaves = go.Figure()
                fig_leaves.add_trace(go.Histogram(
                    x=tree_leaves,
                    nbinsx=min(20, len(set(tree_leaves))),
                    name='Number of Leaves',
                    marker_color='lightgreen'
                ))
                fig_leaves.update_layout(
                    title='Distribution of Leaf Nodes',
                    xaxis_title='Number of Leaves',
                    yaxis_title='Number of Trees',
                    template='plotly_white'
                )
                st.plotly_chart(fig_leaves, use_container_width=True)

            # ========================================
            # 4. FEATURE IMPORTANCE ANALYSIS
            # ========================================
            st.subheader("🎯 Feature Importance Analysis")

            if X.shape[1] <= 20:  # Only show for reasonable number of features
                feature_importance = model.feature_importances_
                
                # Prepare feature names
                if uploaded_file is not None and 'feature_cols' in locals():
                    feature_names = feature_cols
                else:
                    feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
                
                # Create importance dataframe and sort
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=True)
                
                # Feature importance bar chart
                fig_importance = go.Figure()
                fig_importance.add_trace(go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker_color='skyblue',
                    name='Feature Importance'
                ))
                fig_importance.update_layout(
                    title='Feature Importance in Random Forest',
                    xaxis_title='Importance Score',
                    yaxis_title='Features',
                    template='plotly_white',
                    height=max(400, len(feature_names) * 30)
                )
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Feature importance across individual trees
                if model.n_estimators <= 50:  # Only for smaller forests
                    st.subheader("🌲 Feature Importance Across Individual Trees")
                    
                    # Get importance from each tree
                    tree_importances = np.array([tree.feature_importances_ for tree in model.estimators_])
                    
                    # Create heatmap data
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=tree_importances.T,
                        x=[f'Tree {i+1}' for i in range(model.n_estimators)],
                        y=feature_names,
                        colorscale='Viridis',
                        colorbar=dict(title="Importance")
                    ))
                    
                    fig_heatmap.update_layout(
                        title='Feature Importance Across All Trees',
                        xaxis_title='Trees',
                        yaxis_title='Features',
                        template='plotly_white',
                        height=max(400, len(feature_names) * 25)
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    st.info("💡 This heatmap shows how different trees in the forest rely on different features, demonstrating the diversity of the ensemble.")

            # ========================================
            # 5. OUT-OF-BAG (OOB) ANALYSIS
            # ========================================
            if hasattr(model, 'oob_score_') and model.oob_score_ is not None:
                st.subheader("📈 Out-of-Bag (OOB) Analysis")
                
                col_oob1, col_oob2 = st.columns(2)
                
                with col_oob1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>OOB Score</h3>
                        <h2 style="color: green;">{model.oob_score_:.4f}</h2>
                        <p>Out-of-bag accuracy</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_oob2:
                    st.markdown("""
                    <div class="info-box">
                        <h4>💡 About OOB Score</h4>
                        <p>Out-of-bag score uses samples not selected during bootstrap sampling to evaluate each tree, providing an unbiased estimate of the forest's performance without needing a separate validation set.</p>
                    </div>
                    """, unsafe_allow_html=True)

            # ========================================
            # 6. PREDICTION CONSENSUS ANALYSIS
            # ========================================
            if len(X_test) <= 1000:  # Only for reasonable dataset sizes
                st.subheader("🗳️ Prediction Consensus Analysis")
                
                # Get predictions from all trees
                if problem_type == "Classification":
                    # For classification, get class probabilities from all trees
                    all_tree_probs = np.array([tree.predict_proba(X_test) for tree in model.estimators_])
                    
                    # Calculate prediction agreement (how many trees agree on the most likely class)
                    ensemble_predictions = np.argmax(np.mean(all_tree_probs, axis=0), axis=1)
                    individual_predictions = np.array([np.argmax(tree_prob, axis=1) for tree_prob in all_tree_probs])
                    
                    # Calculate consensus for each prediction
                    consensus_scores = []
                    for i in range(len(X_test)):
                        votes_for_prediction = np.sum(individual_predictions[:, i] == ensemble_predictions[i])
                        consensus_scores.append(votes_for_prediction / model.n_estimators)
                    
                    consensus_scores = np.array(consensus_scores)
                    
                else:
                    # For regression, analyze prediction variance
                    all_tree_predictions = np.array([tree.predict(X_test) for tree in model.estimators_])
                    consensus_scores = 1 / (1 + np.std(all_tree_predictions, axis=0))  # Higher score = lower variance
                
                # Plot consensus distribution
                fig_consensus = go.Figure()
                fig_consensus.add_trace(go.Histogram(
                    x=consensus_scores,
                    nbinsx=20,
                    name='Consensus Scores',
                    marker_color='lightcoral'
                ))
                
                title_text = 'Prediction Consensus Distribution'
                if problem_type == "Classification":
                    title_text += ' (Fraction of trees agreeing)'
                else:
                    title_text += ' (Inverse of prediction variance)'
                
                fig_consensus.update_layout(
                    title=title_text,
                    xaxis_title='Consensus Score',
                    yaxis_title='Number of Predictions',
                    template='plotly_white'
                )
                st.plotly_chart(fig_consensus, use_container_width=True)
                
                # Show statistics
                col_cons1, col_cons2, col_cons3 = st.columns(3)
                
                with col_cons1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Avg Consensus</h3>
                        <h2 style="color: blue;">{np.mean(consensus_scores):.3f}</h2>
                        <p>Average agreement</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_cons2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Min Consensus</h3>
                        <h2 style="color: red;">{np.min(consensus_scores):.3f}</h2>
                        <p>Lowest agreement</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_cons3:
                    high_consensus = np.sum(consensus_scores > 0.7) / len(consensus_scores)
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>High Consensus</h3>
                        <h2 style="color: green;">{high_consensus:.1%}</h2>
                        <p>Predictions >70% agreement</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Add a separator before continuing with regular metrics
            st.markdown("---")

            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Visualization based on problem type
            if problem_type == "Classification":
                if X.shape[1] == 2:
                    fig = plot_classification_boundary(X_test, y_test, model, 
                                                    f"Random Forest Classification ({n_estimators} trees)")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Classification metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Display metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Accuracy</h3>
                        <h2 style="color: green;">{accuracy:.3f}</h2>
                        <p>Correct predictions</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Precision</h3>
                        <h2 style="color: blue;">{precision:.3f}</h2>
                        <p>True positive rate</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Recall</h3>
                        <h2 style="color: orange;">{recall:.3f}</h2>
                        <p>Sensitivity</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>F1-Score</h3>
                        <h2 style="color: purple;">{f1:.3f}</h2>
                        <p>Harmonic mean</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            else:  # Regression
                fig = plot_regression_line(X_test, y_test, model, "Random Forest Regression")
                st.plotly_chart(fig, use_container_width=True)
                
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Display metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>R² Score</h3>
                        <h2 style="color: green;">{r2:.3f}</h2>
                        <p>Explained variance</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>MSE</h3>
                        <h2 style="color: red;">{mse:.3f}</h2>
                        <p>Mean Squared Error</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>MAE</h3>
                        <h2 style="color: orange;">{mae:.3f}</h2>
                        <p>Mean Absolute Error</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>RMSE</h3>
                        <h2 style="color: blue;">{rmse:.3f}</h2>
                        <p>Root Mean Squared Error</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Feature importance
            if X.shape[1] <= 10:  # Only show for reasonable number of features
                feature_importance = model.feature_importances_
                feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
                
                fig_importance = go.Figure()
                fig_importance.add_trace(go.Bar(
                    x=feature_names,
                    y=feature_importance,
                    name='Feature Importance'
                ))
                fig_importance.update_layout(
                    title='Feature Importance',
                    xaxis_title='Features',
                    yaxis_title='Importance',
                    template='plotly_white'
                )
                st.plotly_chart(fig_importance, use_container_width=True)

    def svm_page():
        """Support Vector Machine Page"""
        st.markdown('<h1 class="algorithm-header">🎯 Support Vector Machine (SVM)</h1>', unsafe_allow_html=True)
        
        # Algorithm Steps
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **📚 How SVM Works:**
        1. **Find Margin** - Look for the widest possible gap between different classes
        2. **Support Vectors** - Identify the data points closest to the decision boundary
        3. **Maximize Margin** - Create the decision boundary that maximizes the margin
        4. **Kernel Trick** - Use kernel functions to handle non-linear data
        5. **Regularization** - Balance between margin size and classification errors
        6. **Final Boundary** - Use the optimal boundary to classify new data
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🔧 Settings")
            
            # Problem type
            problem_type = st.selectbox("Problem Type:", ["Classification", "Regression"])
            
            # Data selection
            if problem_type == "Classification":
                data_type = st.selectbox("Choose Data Type:", ["Linear Separable", "Circles", "Moons", "Multi-class"])
            else:
                data_type = st.selectbox("Choose Data Type:", ["Linear", "Polynomial", "Sine Wave", "Random"])
            
            n_samples = st.slider("Number of Data Points:", 1, 1000, 300)
            noise = st.slider("Noise Level:", 0.0, 0.5, 0.1)
            
            # Generate data
            X, y = generate_sample_data(data_type, n_samples, noise, problem_type.lower())
            
            # Upload custom data option
            uploaded_file = st.file_uploader("📂 Upload Your Own Data (CSV)", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    st.success("✅ Data loaded successfully!")
                    feature_cols = st.multiselect("📊 Select Feature Columns:", numeric_cols, default=numeric_cols[:-1])
                    target_col = st.selectbox("🎯 Select Target Column:", numeric_cols, index=len(numeric_cols)-1)
                    
                    if target_col in feature_cols:
                        st.error("❌ Target column must be different from feature columns.")
                    elif len(feature_cols) == 0:
                        st.error("❌ Please select at least one feature column.")
                    else:
                        X = df[feature_cols].dropna().values
                        y = df[target_col].dropna().values
                        st.success(f"✅ Using features: {', '.join(feature_cols)} and target: {target_col}")
                else:
                    st.error("❌ Data must have at least 2 numeric columns.")
            
            # Model parameters
            st.subheader("🎯 Model Parameters")
            C = st.slider("Regularization (C):", 0.01, 10.0, 1.0)
            kernel = st.selectbox("Kernel:", ["linear", "rbf", "poly", "sigmoid"])
            
            if kernel == "rbf":
                gamma = st.selectbox("Gamma:", ["scale", "auto"])
            elif kernel == "poly":
                degree = st.slider("Polynomial Degree:", 2, 5, 3)
            
            # Train-test split
            test_size = st.slider("Test Size:", 0.1, 0.5, 0.2)
            
            st.info("""
            💡 **Parameter Tips:**
            - **Higher C**: Less regularization, more complex boundary
            - **RBF Kernel**: Good for non-linear data
            - **Linear Kernel**: Good for linear data, faster training
            """)
        
        with col2:
            st.subheader("🎨 Visualization & Results")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Scale features (important for SVM)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if problem_type == "Classification":
                if kernel == "rbf":
                    model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42, probability=True)
                elif kernel == "poly":
                    model = SVC(C=C, kernel=kernel, degree=degree, random_state=42, probability=True)
                else:
                    model = SVC(C=C, kernel=kernel, random_state=42, probability=True)
            else:
                if kernel == "rbf":
                    model = SVR(C=C, kernel=kernel, gamma=gamma)
                elif kernel == "poly":
                    model = SVR(C=C, kernel=kernel, degree=degree)
                else:
                    model = SVR(C=C, kernel=kernel)
            
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Visualization based on problem type
            if problem_type == "Classification":
                if X.shape[1] == 2:
                    fig = plot_classification_boundary(X_test_scaled, y_test, model, 
                                                    f"SVM Classification (C={C}, kernel={kernel})")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Classification metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Display metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Accuracy</h3>
                        <h2 style="color: green;">{accuracy:.3f}</h2>
                        <p>Correct predictions</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Precision</h3>
                        <h2 style="color: blue;">{precision:.3f}</h2>
                        <p>True positive rate</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Recall</h3>
                        <h2 style="color: orange;">{recall:.3f}</h2>
                        <p>Sensitivity</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>F1-Score</h3>
                        <h2 style="color: purple;">{f1:.3f}</h2>
                        <p>Harmonic mean</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            else:  # Regression
                fig = plot_regression_line(X_test_scaled, y_test, model, "SVM Regression")
                st.plotly_chart(fig, use_container_width=True)
                
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Display metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>R² Score</h3>
                        <h2 style="color: green;">{r2:.3f}</h2>
                        <p>Explained variance</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>MSE</h3>
                        <h2 style="color: red;">{mse:.3f}</h2>
                        <p>Mean Squared Error</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>MAE</h3>
                        <h2 style="color: orange;">{mae:.3f}</h2>
                        <p>Mean Absolute Error</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>RMSE</h3>
                        <h2 style="color: blue;">{rmse:.3f}</h2>
                        <p>Root Mean Squared Error</p>
                    </div>
                    """, unsafe_allow_html=True)

    def knn_page():
        """K-Nearest Neighbors Page"""
        st.markdown('<h1 class="algorithm-header">👥 K-Nearest Neighbors (KNN)</h1>', unsafe_allow_html=True)
        
        # Algorithm Steps
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **📚 How KNN Works:**
        1. **Choose K** - Decide how many neighbors to consider
        2. **Calculate Distance** - Find the distance from new point to all training points
        3. **Find Neighbors** - Select the K closest points
        4. **Vote/Average** - For classification: majority vote, for regression: average
        5. **Make Prediction** - Use the result from step 4 as the final prediction
        6. **No Training** - KNN is a "lazy" algorithm - it doesn't learn during training
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🔧 Settings")
            
            # Problem type
            problem_type = st.selectbox("Problem Type:", ["Classification", "Regression"])
            
            # Data selection
            if problem_type == "Classification":
                data_type = st.selectbox("Choose Data Type:", ["Linear Separable", "Circles", "Moons", "Multi-class"])
            else:
                data_type = st.selectbox("Choose Data Type:", ["Linear", "Polynomial", "Sine Wave", "Random"])
            
            n_samples = st.slider("Number of Data Points:", 1, 1000, 300)
            noise = st.slider("Noise Level:", 0.0, 0.5, 0.1)
            
            # Generate data
            X, y = generate_sample_data(data_type, n_samples, noise, problem_type.lower())
            
            # Upload custom data option
            uploaded_file = st.file_uploader("📂 Upload Your Own Data (CSV)", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    st.success("✅ Data loaded successfully!")
                    feature_cols = st.multiselect("📊 Select Feature Columns:", numeric_cols, default=numeric_cols[:-1])
                    target_col = st.selectbox("🎯 Select Target Column:", numeric_cols, index=len(numeric_cols)-1)
                    
                    if target_col in feature_cols:
                        st.error("❌ Target column must be different from feature columns.")
                    elif len(feature_cols) == 0:
                        st.error("❌ Please select at least one feature column.")
                    else:
                        X = df[feature_cols].dropna().values
                        y = df[target_col].dropna().values
                        st.success(f"✅ Using features: {', '.join(feature_cols)} and target: {target_col}")
                else:
                    st.error("❌ Data must have at least 2 numeric columns.")
            
            # Model parameters
            st.subheader("🎯 Model Parameters")
            n_neighbors = st.slider("Number of Neighbors (K):", 1, 20, 5)
            weights = st.selectbox("Weights:", ["uniform", "distance"])
            metric = st.selectbox("Distance Metric:", ["euclidean", "manhattan", "minkowski"])
            
            # Train-test split
            test_size = st.slider("Test Size:", 0.1, 0.5, 0.2)
            
            st.info("""
            💡 **Parameter Tips:**
            - **Small K**: More sensitive to noise, complex boundaries
            - **Large K**: Smoother boundaries, less sensitive to noise
            - **Distance weights**: Closer neighbors have more influence
            """)
        
        with col2:
            st.subheader("🎨 Visualization & Results")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Scale features (important for KNN)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if problem_type == "Classification":
                model = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    metric=metric
                )
            else:
                model = KNeighborsRegressor(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    metric=metric
                )
            
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Visualization based on problem type
            if problem_type == "Classification":
                if X.shape[1] == 2:
                    fig = plot_classification_boundary(X_test_scaled, y_test, model, 
                                                    f"KNN Classification (K={n_neighbors})")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Classification metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Display metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Accuracy</h3>
                        <h2 style="color: green;">{accuracy:.3f}</h2>
                        <p>Correct predictions</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Precision</h3>
                        <h2 style="color: blue;"
                        <h2 style="color: blue;">{precision:.3f}</h2>
                        <p>True positive rate</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Recall</h3>
                        <h2 style="color: orange;">{recall:.3f}</h2>
                        <p>Sensitivity</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_m4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>F1-Score</h3>
                        <h2 style="color: purple;">{f1:.3f}</h2>
                        <p>Harmonic mean</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Confusion Matrix
                cm_fig = plot_confusion_matrix(y_test, y_pred, "Confusion Matrix")
                st.plotly_chart(cm_fig, use_container_width=True)

            else:
                # Regression Visualization
                fig = plot_regression_line(X_test_scaled, y_test, model, f"KNN Regression (K={n_neighbors})")
                st.plotly_chart(fig, use_container_width=True)

                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>R² Score</h3>
                        <h2 style="color: green;">{r2:.3f}</h2>
                        <p>Explained variance</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>MSE</h3>
                        <h2 style="color: red;">{mse:.3f}</h2>
                        <p>Mean Squared Error</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>MAE</h3>
                        <h2 style="color: orange;">{mae:.3f}</h2>
                        <p>Mean Absolute Error</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_m4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>RMSE</h3>
                        <h2 style="color: blue;">{rmse:.3f}</h2>
                        <p>Root Mean Squared Error</p>
                    </div>
                    """, unsafe_allow_html=True)
    # Sidebar Navigation
    st.sidebar.title("🔍 Select Algorithm")
    page = st.sidebar.radio("Choose one:", [
        "🏠 Home",
        "📊 Logistic Regression",
        "📈 Linear Regression",
        "🌳 Decision Tree",
        "🌲 Random Forest",
        "🎯 Support Vector Machine (SVM)",
        "👥 K-Nearest Neighbors (KNN)"
    ])

    if page == "🏠 Home":
        st.markdown('<h1 class="main-header">Welcome to Supervised Learning Visual Classifier</h1>', unsafe_allow_html=True)
        st.markdown("""
        This interactive app lets students learn how different **supervised machine learning algorithms** work using:
        - 🎯 Step-by-step algorithm descriptions  
        - 📊 Dynamic visualizations and decision boundaries  
        - 🧪 Uploadable or auto-generated datasets  
        - 📈 Live metrics like accuracy, F1-score, R², etc.  

        👉 Use the sidebar to choose an algorithm to begin!
        """)
    elif page == "📊 Logistic Regression":
        logistic_regression_page()
    elif page == "📈 Linear Regression":
        linear_regression_page()
    elif page == "🌳 Decision Tree":
        decision_tree_page()
    elif page == "🌲 Random Forest":
        random_forest_page()
    elif page == "🎯 Support Vector Machine (SVM)":
        svm_page()
    elif page == "👥 K-Nearest Neighbors (KNN)":
        knn_page()

def supervised_main():
    main()
                                