import os
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


USER_FILE = "users.csv"
REPORTS_DIR = "reports"

if not os.path.exists(USER_FILE):
    pd.DataFrame({"username": ["admin"], "password": ["admin123"]}).to_csv(USER_FILE, index=False)

os.makedirs(REPORTS_DIR, exist_ok=True)


def load_users():
    return pd.read_csv(USER_FILE)


def register_user(username, password):
    users = load_users()
    username = username.strip()
    password = password.strip()
    users["username"] = users["username"].astype(str).str.strip()
    if username in users["username"].values:
        return False
    users.loc[len(users)] = [username, password]
    users.to_csv(USER_FILE, index=False)
    return True


def authenticate(username, password):
    users = load_users()
    users["username"] = users["username"].astype(str).str.strip()
    users["password"] = users["password"].astype(str).str.strip()
    username = username.strip()
    password = password.strip()
    match = users[(users["username"] == username) & (users["password"] == password)]
    return len(match) > 0


def calculate_quality_score(dataframe):
    total_cells = dataframe.shape[0] * dataframe.shape[1]
    if total_cells == 0:
        return 0, 0, total_cells
    missing_cells = int(dataframe.isnull().sum().sum())
    quality_score = round((1 - missing_cells / total_cells) * 100, 2)
    return quality_score, missing_cells, total_cells


def get_dataset_metrics(dataframe):
    quality_score, missing_cells, _ = calculate_quality_score(dataframe)
    return {
        "Rows": dataframe.shape[0],
        "Columns": dataframe.shape[1],
        "Missing Values": missing_cells,
        "Duplicate Rows": int(dataframe.duplicated().sum()),
        "Quality Score": quality_score,
    }


def detect_outliers(dataframe):
    outlier_df = dataframe.select_dtypes(include=["number"]).copy()
    if "PassengerId" in outlier_df.columns:
        outlier_df = outlier_df.drop(columns=["PassengerId"])
    if len(outlier_df.columns) <= 1:
        return 0, 0, None
    outlier_df = outlier_df.dropna()
    if len(outlier_df) == 0:
        return 0, 0, None
    iso = IsolationForest(contamination=0.05, random_state=42)
    labels = iso.fit_predict(outlier_df)
    outlier_count = int((labels == -1).sum())
    outlier_percentage = round((outlier_count / len(outlier_df)) * 100, 2)
    outlier_df["Outlier"] = labels
    return outlier_count, outlier_percentage, outlier_df


def compute_reliability_scores(quality_score, missing_cells, total_cells, duplicates, row_count, outlier_percentage):
    missing_score = max(0, round(100 - ((missing_cells / total_cells) * 100), 2)) if total_cells else 0
    duplicate_score = 100 if duplicates == 0 else max(0, round(100 - ((duplicates / row_count) * 100), 2))
    outlier_score = max(0, round(100 - outlier_percentage, 2))
    overall_score = round((missing_score + duplicate_score + outlier_score) / 3, 2)
    if overall_score >= 90:
        grade = "A"
    elif overall_score >= 80:
        grade = "B"
    elif overall_score >= 70:
        grade = "C"
    else:
        grade = "D"
    return missing_score, duplicate_score, outlier_score, overall_score, grade


def recommend_target_columns(dataframe):
    candidates = []
    for col in dataframe.columns:
        unique_count = dataframe[col].nunique()
        if unique_count >= 2 and unique_count <= 20 and "id" not in col.lower():
            candidates.append(col)
    return candidates


def generate_research_recommendations(quality_score, missing_cells, duplicates, outlier_count, dataframe):
    recommendations = []
    if quality_score >= 90:
        recommendations.append("Dataset quality is high and suitable for supervised learning experiments.")
    else:
        recommendations.append("Dataset requires additional preprocessing before reliable model deployment.")
    if missing_cells > 0:
        worst_missing_col = dataframe.isnull().sum().idxmax()
        worst_missing_count = int(dataframe.isnull().sum().max())
        recommendations.append(f"Column '{worst_missing_col}' has the highest missingness with {worst_missing_count} missing values.")
    if duplicates == 0:
        recommendations.append("No duplicate rows were found, which improves dataset reliability.")
    else:
        recommendations.append("Duplicate rows should be reviewed or removed before training.")
    if outlier_count > 0:
        recommendations.append(f"{outlier_count} anomalous records were detected and should be reviewed before production use.")
    recommendations.append("Benchmark multiple models before selecting the final production model.")
    recommendations.append("Use SHAP explainability to understand how individual features influence model behavior.")
    return recommendations


def create_research_abstract(rows, cols, quality_score, overall_score, grade, outlier_count, best_model_name=None, best_accuracy=None, top_shap_feature=None):
    abstract = (
        f"This study presents an automated dataset intelligence analysis over a dataset containing {rows} observations and {cols} features. "
        f"The platform computed a dataset quality score of {quality_score}% and an overall reliability score of {overall_score}, corresponding to grade {grade}. "
        f"Isolation Forest analysis identified {outlier_count} anomalous observations."
    )
    if best_model_name is not None and best_accuracy is not None:
        abstract += f" Model benchmarking selected {best_model_name} as the strongest baseline model with an accuracy of {best_accuracy:.2%}."
    if top_shap_feature is not None:
        abstract += f" SHAP explainability analysis identified {top_shap_feature} as the most influential feature contributing to model behavior."
    abstract += " The system demonstrates an end-to-end explainable AI workflow for dataset profiling, reliability assessment, anomaly detection, model evaluation, and research-style reporting."
    return abstract


def generate_pdf_report(output_path, rows, cols, quality_score, missing_cells, duplicates, outlier_count, outlier_percentage, missing_score, duplicate_score, outlier_score, overall_score, grade, research_abstract, research_recommendations, benchmark_df=None, feature_importance_df=None, shap_importance_df=None):
    doc = SimpleDocTemplate(output_path)
    styles = getSampleStyleSheet()
    content = []
    content.append(Paragraph("AutoInsight AI Research Report", styles["Title"]))
    content.append(Spacer(1, 20))
    content.append(Paragraph("Research Abstract", styles["Heading1"]))
    content.append(Paragraph(research_abstract, styles["Normal"]))
    content.append(Spacer(1, 15))
    content.append(Paragraph("Executive Summary", styles["Heading1"]))
    content.append(Paragraph(f"This dataset contains {rows} observations and {cols} features. The dataset quality score is {quality_score}%. The overall reliability score is {overall_score}, with grade {grade}. Isolation Forest detected {outlier_count} anomalous observations.", styles["Normal"]))
    content.append(Spacer(1, 15))
    content.append(Paragraph("Dataset Profiling Results", styles["Heading1"]))
    stats = [f"Rows: {rows}", f"Columns: {cols}", f"Quality Score: {quality_score}%", f"Missing Values: {missing_cells}", f"Duplicate Rows: {duplicates}", f"Outliers Detected: {outlier_count}", f"Outlier Percentage: {outlier_percentage}%"]
    for item in stats:
        content.append(Paragraph(item, styles["Normal"]))
    content.append(Spacer(1, 15))
    content.append(Paragraph("Dataset Reliability Index", styles["Heading1"]))
    reliability_items = [f"Missing Data Score: {missing_score}", f"Duplicate Score: {duplicate_score}", f"Outlier Score: {outlier_score}", f"Overall Reliability Score: {overall_score}", f"Reliability Grade: {grade}"]
    for item in reliability_items:
        content.append(Paragraph(item, styles["Normal"]))
    content.append(Spacer(1, 15))
    content.append(Paragraph("Model Benchmark Results", styles["Heading1"]))
    if benchmark_df is not None:
        for _, row in benchmark_df.iterrows():
            content.append(Paragraph(f"{row['Model']} : {row['Accuracy']}%", styles["Normal"]))
    else:
        content.append(Paragraph("Model benchmarking was not performed before report generation.", styles["Normal"]))
    content.append(Spacer(1, 15))
    content.append(Paragraph("Feature Importance", styles["Heading1"]))
    if feature_importance_df is not None:
        for _, row in feature_importance_df.head(5).iterrows():
            content.append(Paragraph(f"{row['Feature']} : {round(row['Importance'], 4)}", styles["Normal"]))
    else:
        content.append(Paragraph("Feature importance was not available.", styles["Normal"]))
    content.append(Spacer(1, 15))
    content.append(Paragraph("Explainable AI Findings - SHAP", styles["Heading1"]))
    if shap_importance_df is not None:
        for _, row in shap_importance_df.head(5).iterrows():
            content.append(Paragraph(f"{row['Feature']} : {round(row['SHAP Importance'], 4)}", styles["Normal"]))
    else:
        content.append(Paragraph("SHAP explanation was not available.", styles["Normal"]))
    content.append(Spacer(1, 15))
    content.append(Paragraph("Research Recommendations", styles["Heading1"]))
    for item in research_recommendations:
        content.append(Paragraph("• " + item, styles["Normal"]))
    content.append(Spacer(1, 20))
    content.append(Paragraph("Generated by AutoInsight AI - Explainable Dataset Intelligence Platform", styles["Italic"]))
    doc.build(content)
    return output_path


st.set_page_config(page_title="AutoInsight AI", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #0f766e 100%); color: white;}
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.hero-box {background: linear-gradient(135deg, #1e293b, #0f766e); padding: 35px; border-radius: 20px; margin-bottom: 25px; box-shadow: 0px 8px 30px rgba(0,0,0,0.35);}
.hero-title {font-size: 44px; font-weight: 900; color: white;}
.hero-subtitle {font-size: 18px; color: #d1fae5; margin-top: 8px;}
.login-box {max-width: 460px; margin: auto; margin-top: 80px; background-color: rgba(30,41,59,0.96); padding: 35px; border-radius: 22px; box-shadow: 0px 8px 35px rgba(0,0,0,0.45); border: 1px solid #334155;}
[data-testid="stMetricValue"] {color: #34d399;}
[data-testid="stSidebar"] {background-color: #020617;}
</style>
""", unsafe_allow_html=True)

for key, value in {
    "authenticated": False,
    "username": "",
    "benchmark_results": None,
    "feature_importance": None,
    "shap_importance": None,
    "best_model_name": None,
    "best_accuracy": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def login_page():
    st.markdown("""
    <div class="login-box">
        <h1 style="color:white;">🔐 AutoInsight AI</h1>
        <p style="color:#cbd5e1;">Login or create your own account.</p>
    </div>
    """, unsafe_allow_html=True)
    option = st.radio("Select Option", ["Login", "Register"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if option == "Login":
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username.strip()
                st.rerun()
            else:
                st.error("Invalid username or password.")
    else:
        confirm_password = st.text_input("Confirm Password", type="password")
        if st.button("Create Account"):
            if username.strip() == "":
                st.error("Username required.")
            elif password.strip() == "":
                st.error("Password required.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                if register_user(username.strip(), password.strip()):
                    st.success("Account created successfully. Please login.")
                else:
                    st.error("Username already exists.")


if not st.session_state["authenticated"]:
    login_page()
    st.stop()

with st.sidebar:
    st.title("⚙️ AutoInsight AI")
    current_user = st.session_state.get("username", "Unknown")
    st.write(f"👤 Logged in as: **{current_user}**")
    users = load_users()
    st.write(f"👥 Registered Users: **{len(users)}**")
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = ""
        st.rerun()
    st.markdown("---")
    st.write("### Platform Modules")
    st.write("📊 Dataset Profiling")
    st.write("📈 Visual Analytics")
    st.write("🚨 Outlier Detection")
    st.write("📊 Reliability Index")
    st.write("🤖 Model Benchmarking")
    st.write("🧠 SHAP Explainability")
    st.write("📄 PDF Research Report")
    st.write("💬 Dataset Chat")
    if current_user == "admin":
        st.markdown("---")
        st.write("### 🛡️ Admin Panel")
        st.write("Admin dashboard enabled.")

st.markdown("""
<div class="hero-box">
    <div class="hero-title">🤖 AutoInsight AI</div>
    <div class="hero-subtitle">
        Explainable Dataset Intelligence Platform for automated profiling,
        anomaly detection, model benchmarking, SHAP explainability, user-specific report history, and research reporting.
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
    tab_names = ["📊 Overview", "📈 Visual Analytics", "🚨 Reliability & Outliers", "🤖 ML Lab", "📄 Report", "💬 Chat", "🧬 Dataset Comparison"]
    if st.session_state["username"] == "admin":
        tab_names.append("🛡️ Admin Dashboard")
    tabs = st.tabs(tab_names)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = tabs[:7]
    tab8 = tabs[7] if st.session_state["username"] == "admin" else None

    duplicates = int(df.duplicated().sum())
    quality_score, missing_cells, total_cells = calculate_quality_score(df)
    numeric_cols = df.select_dtypes(include=["number"]).columns
    missing_df = pd.DataFrame({"Column": df.columns, "Missing Values": df.isnull().sum().values})
    outlier_count, outlier_percentage, outlier_df = detect_outliers(df)
    missing_score, duplicate_score, outlier_score, overall_score, grade = compute_reliability_scores(quality_score, missing_cells, total_cells, duplicates, df.shape[0], outlier_percentage)
    top_shap_feature = st.session_state["shap_importance"].iloc[0]["Feature"] if st.session_state["shap_importance"] is not None else None
    research_abstract = create_research_abstract(df.shape[0], df.shape[1], quality_score, overall_score, grade, outlier_count, st.session_state["best_model_name"], st.session_state["best_accuracy"], top_shap_feature)
    research_recommendations = generate_research_recommendations(quality_score, missing_cells, duplicates, outlier_count, df)

    with tab1:
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.subheader("📏 Dataset Shape")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", missing_cells)
        col4.metric("Quality Score", f"{quality_score}%")
        st.subheader("📝 Column Names")
        st.write(list(df.columns))
        st.subheader("🔍 Data Types")
        dtype_df = pd.DataFrame({"Column": df.columns, "Data Type": df.dtypes.astype(str)})
        st.dataframe(dtype_df, use_container_width=True)
        st.subheader("❌ Missing Values")
        st.dataframe(missing_df, use_container_width=True)
        st.subheader("♻️ Duplicate Rows")
        st.metric("Duplicates", duplicates)
        st.subheader("🧠 AI Insights")
        insights = []
        if missing_cells > 0:
            insights.append(f"The dataset contains {missing_cells} missing values.")
        if duplicates > 0:
            insights.append(f"The dataset contains {duplicates} duplicate rows.")
        if df.shape[0] > 500:
            insights.append("Dataset size is sufficient for machine learning experiments.")
        insights.append(f"The dataset contains {len(numeric_cols)} numerical features.")
        for item in insights:
            st.write("✅", item)

    with tab2:
        st.subheader("📊 Missing Values Visualization")
        missing_chart = missing_df[missing_df["Missing Values"] > 0]
        if len(missing_chart) > 0:
            fig = px.bar(missing_chart, x="Column", y="Missing Values", title="Missing Values by Column")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No missing values found.")
        st.subheader("📈 Numerical Feature Distribution")
        numeric_columns = df.select_dtypes(include=["number"]).columns
        if len(numeric_columns) > 0:
            selected_column = st.selectbox("Select a numerical column", numeric_columns)
            fig = px.histogram(df, x=selected_column, title=f"Distribution of {selected_column}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numerical columns available for histogram.")
        st.subheader("🔥 Correlation Heatmap")
        numeric_df = df.select_dtypes(include=["number"])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numerical columns for correlation heatmap.")

    with tab3:
        st.subheader("🚨 Outlier Detection")
        if outlier_df is not None:
            col_a, col_b = st.columns(2)
            col_a.metric("Detected Outliers", outlier_count)
            col_b.metric("Outlier Percentage", f"{outlier_percentage}%")
            if "Age" in outlier_df.columns and "Fare" in outlier_df.columns:
                fig = px.scatter(outlier_df, x="Age", y="Fare", color="Outlier", title="Age vs Fare Outlier Analysis")
                st.plotly_chart(fig, use_container_width=True)
            else:
                numeric_plot_cols = [col for col in outlier_df.columns if col != "Outlier"]
                if len(numeric_plot_cols) >= 2:
                    fig = px.scatter(outlier_df, x=numeric_plot_cols[0], y=numeric_plot_cols[1], color="Outlier", title="Outlier Detection Visualization")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numerical data for outlier detection.")
        st.subheader("📊 Dataset Reliability Index")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Missing Data Score", missing_score)
        r2.metric("Duplicate Score", duplicate_score)
        r3.metric("Outlier Score", outlier_score)
        r4.metric("Overall Reliability", overall_score)
        st.success(f"Dataset Reliability Grade: {grade}")
        st.subheader("🎓 Dataset Report Card")
        report_card = pd.DataFrame({"Category": ["Completeness", "Consistency", "Anomaly Risk", "Model Readiness"], "Grade": ["A" if quality_score >= 90 else "B", "A" if duplicates == 0 else "B", "A" if outlier_percentage < 5 else "B", "A" if overall_score >= 90 else "B"]})
        st.dataframe(report_card, use_container_width=True)
        st.subheader("🔬 Research Findings")
        st.write(f"• Dataset contains {df.shape[0]} records and {df.shape[1]} features.")
        st.write(f"• Dataset quality score is {quality_score}%.")
        st.write(f"• Dataset reliability grade is {grade}.")
        st.write(f"• Dataset contains {missing_cells} missing values, which may require preprocessing." if missing_cells > 0 else "• Dataset has no missing values.")
        st.write(f"• Dataset contains {duplicates} duplicate rows." if duplicates > 0 else "• Dataset has no duplicate rows.")
        st.write(f"• Isolation Forest detected {outlier_count} anomalous observations, representing approximately {outlier_percentage}% of complete numeric records.")

    with tab4:
        st.subheader("🎯 Auto ML Recommendations")
        target_candidates = recommend_target_columns(df)
        if len(target_candidates) > 0:
            st.write("Recommended target columns:")
            for col in target_candidates:
                st.write("✅", col)
            st.info(f"Suggested Target: {target_candidates[0]}")
        else:
            st.warning("No strong target candidate detected automatically.")
        st.subheader("🤖 ML Problem Detection")
        target_column = st.selectbox("Select Target Column", df.columns)
        unique_values = df[target_column].nunique()
        if unique_values <= 20:
            problem_type = "Classification"
            recommended_models = ["Logistic Regression", "Random Forest", "XGBoost", "Decision Tree"]
        else:
            problem_type = "Regression"
            recommended_models = ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor"]
        st.success(f"Detected Problem Type: {problem_type}")
        st.subheader("Recommended Models")
        for model_name in recommended_models:
            st.write("✅", model_name)
        st.subheader("🚀 Train & Benchmark Models")
        if st.button("Train Models"):
            if problem_type != "Classification":
                st.error("This current demo benchmarking engine supports classification targets. Please select a target like Survived.")
            else:
                try:
                    train_df = df.copy().select_dtypes(include=["number"]).dropna()
                    if target_column not in train_df.columns:
                        st.error("Target column must be numeric for this demo. Please select a numeric target like Survived.")
                    else:
                        X = train_df.drop(columns=[target_column])
                        y = train_df[target_column]
                        if "PassengerId" in X.columns:
                            X = X.drop(columns=["PassengerId"])
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        models = {"Logistic Regression": LogisticRegression(max_iter=1000), "Decision Tree": DecisionTreeClassifier(random_state=42), "Random Forest": RandomForestClassifier(random_state=42), "Gradient Boosting": GradientBoostingClassifier(random_state=42)}
                        results = []
                        best_model = None
                        best_model_name = None
                        best_accuracy = 0
                        for model_name, model in models.items():
                            model.fit(X_train, y_train)
                            predictions = model.predict(X_test)
                            accuracy = accuracy_score(y_test, predictions)
                            results.append({"Model": model_name, "Accuracy": round(accuracy * 100, 2)})
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_model = model
                                best_model_name = model_name
                        results_df = pd.DataFrame(results)
                        st.session_state["benchmark_results"] = results_df
                        st.session_state["best_model_name"] = best_model_name
                        st.session_state["best_accuracy"] = best_accuracy
                        st.subheader("🏆 Model Benchmarking")
                        st.dataframe(results_df, use_container_width=True)
                        fig = px.bar(results_df, x="Model", y="Accuracy", title="Model Performance Comparison")
                        st.plotly_chart(fig, use_container_width=True)
                        st.success(f"Best Model: {best_model_name} with Accuracy: {best_accuracy:.2%}")
                        rf_model = RandomForestClassifier(random_state=42)
                        rf_model.fit(X_train, y_train)
                        feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": rf_model.feature_importances_}).sort_values(by="Importance", ascending=False)
                        st.session_state["feature_importance"] = feature_importance
                        st.subheader("🔥 Feature Importance")
                        st.dataframe(feature_importance, use_container_width=True)
                        fig = px.bar(feature_importance, x="Feature", y="Importance", title="Feature Importance")
                        st.plotly_chart(fig, use_container_width=True)
                        st.subheader("🧠 Explainable AI (SHAP)")
                        try:
                            explainer = shap.TreeExplainer(rf_model)
                            shap_values = explainer.shap_values(X)
                            shap_array = shap_values
                            if isinstance(shap_values, list):
                                shap_array = shap_values[1]
                            if len(np.array(shap_array).shape) == 3:
                                shap_array = np.array(shap_array)[:, :, 1]
                            shap_importance = pd.DataFrame({"Feature": X.columns, "SHAP Importance": np.abs(shap_array).mean(axis=0)}).sort_values(by="SHAP Importance", ascending=False)
                            st.session_state["shap_importance"] = shap_importance
                            st.dataframe(shap_importance, use_container_width=True)
                            fig = px.bar(shap_importance, x="Feature", y="SHAP Importance", title="SHAP Feature Contributions")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"SHAP explanation unavailable: {e}")
                except Exception as e:
                    st.error(str(e))

    top_shap_feature = st.session_state["shap_importance"].iloc[0]["Feature"] if st.session_state["shap_importance"] is not None else None
    research_abstract = create_research_abstract(df.shape[0], df.shape[1], quality_score, overall_score, grade, outlier_count, st.session_state["best_model_name"], st.session_state["best_accuracy"], top_shap_feature)

    with tab5:
        st.subheader("📝 Research Abstract")
        st.write(research_abstract)
        st.subheader("🧪 AI Research Recommendations")
        for rec in research_recommendations:
            st.write("🔹", rec)
        st.subheader("📄 Generate Research Report")
        if st.button("Generate PDF Report"):
            username = st.session_state["username"]
            user_report_dir = os.path.join(REPORTS_DIR, username)
            os.makedirs(user_report_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_path = os.path.join(user_report_dir, f"research_report_{timestamp}.pdf")
            pdf_file = generate_pdf_report(pdf_path, df.shape[0], df.shape[1], quality_score, missing_cells, duplicates, outlier_count, outlier_percentage, missing_score, duplicate_score, outlier_score, overall_score, grade, research_abstract, research_recommendations, st.session_state["benchmark_results"], st.session_state["feature_importance"], st.session_state["shap_importance"])
            with open(pdf_file, "rb") as file:
                st.download_button(label="⬇ Download Research Report", data=file, file_name=os.path.basename(pdf_file), mime="application/pdf")
            st.success("Report saved to your report history.")
        st.subheader("🗂️ My Report History")
        username = st.session_state["username"]
        user_report_dir = os.path.join(REPORTS_DIR, username)
        os.makedirs(user_report_dir, exist_ok=True)
        reports = sorted([file for file in os.listdir(user_report_dir) if file.endswith(".pdf")], reverse=True)
        if len(reports) == 0:
            st.info("No saved reports yet.")
        else:
            for report in reports:
                report_path = os.path.join(user_report_dir, report)
                with open(report_path, "rb") as file:
                    st.download_button(label=f"⬇ {report}", data=file, file_name=report, mime="application/pdf")

    with tab6:
        st.subheader("💬 Dataset Chat Assistant")
        user_question = st.text_input("Ask a question about your dataset")
        if user_question:
            question = user_question.lower()
            if "row" in question:
                st.write(f"The dataset has {df.shape[0]} rows.")
            elif "column" in question:
                st.write(f"The dataset has {df.shape[1]} columns.")
            elif "missing" in question:
                st.write(f"The column with the most missing values is '{df.isnull().sum().idxmax()}' with {df.isnull().sum().max()} missing values.")
            elif "duplicate" in question:
                st.write(f"The dataset has {df.duplicated().sum()} duplicate rows.")
            elif "quality" in question:
                st.write(f"The dataset quality score is {quality_score}%.")
            elif "reliability" in question or "grade" in question:
                st.write(f"The dataset reliability score is {overall_score}, with grade {grade}.")
            elif "outlier" in question or "anomaly" in question:
                st.write(f"Isolation Forest detected {outlier_count} anomalous observations, approximately {outlier_percentage}% of complete numeric records.")
            elif "shap" in question or "explain" in question:
                shap_df = st.session_state["shap_importance"]
                if shap_df is not None:
                    st.write(f"The most important SHAP feature is {shap_df.iloc[0]['Feature']} with SHAP importance {round(shap_df.iloc[0]['SHAP Importance'], 4)}.")
                else:
                    st.write("Please train the model first to generate SHAP explanations.")
            elif "abstract" in question:
                st.write(research_abstract)
            elif "recommendation" in question:
                for rec in research_recommendations:
                    st.write("🔹", rec)
            elif "numerical" in question or "numeric" in question:
                st.write(f"The dataset has {len(numeric_cols)} numerical columns.")
            elif "average" in question or "mean" in question:
                matched_column = None
                for col in df.select_dtypes(include=["number"]).columns:
                    if col.lower() in question:
                        matched_column = col
                        break
                if matched_column:
                    st.write(f"The average value of {matched_column} is {df[matched_column].mean():.2f}.")
                else:
                    st.write("Please mention a numerical column name. Example: What is the average Age?")
            else:
                st.write("I can answer questions about rows, columns, missing values, duplicates, quality score, reliability, outliers, SHAP explanations, abstract, recommendations, numerical columns, and averages.")

    with tab7:
        st.subheader("🧬 Dataset Comparison Dashboard")
        st.write("Upload a second CSV file to compare it with the current dataset.")
        comparison_file = st.file_uploader("Upload comparison CSV file", type=["csv"], key="comparison_file")
        if comparison_file is not None:
            df2 = pd.read_csv(comparison_file)
            metrics_1 = get_dataset_metrics(df)
            metrics_2 = get_dataset_metrics(df2)
            comparison_df = pd.DataFrame({"Metric": list(metrics_1.keys()), "Current Dataset": list(metrics_1.values()), "Comparison Dataset": list(metrics_2.values())})
            st.dataframe(comparison_df, use_container_width=True)
            fig = px.bar(comparison_df, x="Metric", y=["Current Dataset", "Comparison Dataset"], barmode="group", title="Dataset Comparison")
            st.plotly_chart(fig, use_container_width=True)

    if st.session_state["username"] == "admin" and tab8 is not None:
        with tab8:
            st.subheader("🛡️ Admin Dashboard")
            users = load_users()
            total_users = len(users)
            total_reports = 0
            report_rows = []
            if os.path.exists(REPORTS_DIR):
                for user in os.listdir(REPORTS_DIR):
                    user_dir = os.path.join(REPORTS_DIR, user)
                    if os.path.isdir(user_dir):
                        pdf_reports = [file for file in os.listdir(user_dir) if file.endswith(".pdf")]
                        count = len(pdf_reports)
                        total_reports += count
                        report_rows.append({"User": user, "Reports Generated": count})
            c1, c2 = st.columns(2)
            c1.metric("Total Registered Users", total_users)
            c2.metric("Total Saved Reports", total_reports)
            st.subheader("Registered Users")
            st.dataframe(users[["username"]], use_container_width=True)
            st.subheader("Report Activity")
            if len(report_rows) > 0:
                activity_df = pd.DataFrame(report_rows)
                st.dataframe(activity_df, use_container_width=True)
                fig = px.bar(activity_df, x="User", y="Reports Generated", title="Reports Generated per User")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No report activity yet.")
else:
    st.info("Please upload a CSV file to begin.")
