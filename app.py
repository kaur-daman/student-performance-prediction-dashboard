
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans

st.set_page_config(page_title="Student AI Dashboard", layout="wide")

st.title("🎓 AI Student Performance Intelligence System")

# -----------------------------
# DATASET
# -----------------------------
np.random.seed(42)
size = 1200

data = pd.DataFrame({
    "study_hours": np.random.randint(1,10,size),
    "attendance": np.random.randint(50,100,size),
    "sleep_hours": np.random.randint(4,9,size),
    "previous_grade": np.random.randint(40,95,size),
    "assignments_completed": np.random.randint(0,10,size),
    "internet_usage_hours": np.random.randint(1,8,size),
    "sports_hours": np.random.randint(0,5,size),
    "social_media_hours": np.random.randint(0,6,size)
})

score = (
    data["study_hours"]*5 +
    data["attendance"]*0.3 +
    data["previous_grade"]*0.5 +
    data["assignments_completed"]*2 -
    data["social_media_hours"]*2
)

conditions = [
    score > 90,
    (score > 70) & (score <= 90),
    score <= 70
]

choices = ["Excellent","Average","Poor"]

data["performance"] = np.select(conditions, choices, default="Poor")

# -----------------------------
# RISK DETECTION
# -----------------------------
data["risk"] = np.where(data["performance"]=="Poor","High Risk","Low Risk")

# -----------------------------
# MODEL TRAINING
# -----------------------------
X = data.drop(["performance","risk"], axis=1)
y = data["performance"]

model = RandomForestClassifier(n_estimators=200)
model.fit(X,y)

y_pred = model.predict(X)

accuracy = accuracy_score(y,y_pred)

# -----------------------------
# CLUSTERING
# -----------------------------
kmeans = KMeans(n_clusters=3,n_init=10)
data["cluster"] = kmeans.fit_predict(X)

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("⚙ Student Inputs")

study_hours = st.sidebar.slider("Study Hours",0,12,4)
attendance = st.sidebar.slider("Attendance",40,100,75)
sleep_hours = st.sidebar.slider("Sleep Hours",3,10,7)
previous_grade = st.sidebar.slider("Previous Grade",0,100,70)
assignments_completed = st.sidebar.slider("Assignments",0,10,5)
internet_usage_hours = st.sidebar.slider("Internet Usage",0,10,4)
sports_hours = st.sidebar.slider("Sports",0,6,1)
social_media_hours = st.sidebar.slider("Social Media",0,8,3)

input_df = pd.DataFrame({
    "study_hours":[study_hours],
    "attendance":[attendance],
    "sleep_hours":[sleep_hours],
    "previous_grade":[previous_grade],
    "assignments_completed":[assignments_completed],
    "internet_usage_hours":[internet_usage_hours],
    "sports_hours":[sports_hours],
    "social_media_hours":[social_media_hours]
})

prediction = model.predict(input_df)[0]
confidence = model.predict_proba(input_df).max()

# -----------------------------
# METRICS
# -----------------------------
c1,c2,c3 = st.columns(3)

c1.metric("Predicted Performance",prediction)
c2.metric("Confidence",f"{round(confidence*100,2)}%")
c3.metric("Model Accuracy",f"{round(accuracy*100,2)}%")

st.divider()

# =====================================================
# STATIC GRAPHS
# =====================================================

st.subheader("📊 Static Graphs")

col1,col2 = st.columns(2)

with col1:
    fig1 = px.histogram(
        data,
        x="performance",
        title="Student Performance Distribution"
    )
    st.plotly_chart(fig1,use_container_width=True)

# -----------------------------
# DYNAMIC RISK PIE CHART
# -----------------------------

filtered_data = data[
    (data["study_hours"] >= study_hours-2) &
    (data["attendance"] >= attendance-10) &
    (data["sleep_hours"] >= sleep_hours-1)
]

with col2:
    fig2 = px.pie(
        filtered_data,
        names="risk",
        title="Dynamic Risk Distribution"
    )
    st.plotly_chart(fig2,use_container_width=True)

# =====================================================
# DYNAMIC COMPARISON
# =====================================================

st.subheader("📈 Dynamic Comparison")

col3,col4 = st.columns(2)

with col3:
    fig3 = px.scatter(
        data,
        x="study_hours",
        y="previous_grade",
        color="performance",
        size="attendance",
        title="Study Hours vs Grades"
    )
    st.plotly_chart(fig3,use_container_width=True)

with col4:
    fig4 = px.scatter(
        data,
        x="sleep_hours",
        y="study_hours",
        color="performance",
        size="assignments_completed",
        title="Sleep vs Study Comparison"
    )
    st.plotly_chart(fig4,use_container_width=True)

# =====================================================
# RISK ANALYSIS BOXPLOT
# =====================================================

st.subheader("⚠ Risk Analysis")

fig5 = px.box(
    data,
    x="risk",
    y="social_media_hours",
    color="risk",
    title="Social Media Impact on Risk"
)

st.plotly_chart(fig5,use_container_width=True)

# =====================================================
# MODEL EVALUATION
# =====================================================

st.subheader("🤖 Model Evaluation")

cm = confusion_matrix(y,y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Excellent","Average","Poor"],
    columns=["Excellent","Average","Poor"]
)

fig6 = px.imshow(cm_df,text_auto=True,title="Confusion Matrix")

st.plotly_chart(fig6,use_container_width=True)

# =====================================================
# CLUSTERING
# =====================================================

st.subheader("🧠 Student Segmentation")

fig7 = px.scatter(
    data,
    x="study_hours",
    y="attendance",
    color=data["cluster"].astype(str),
    title="Student Clusters"
)

st.plotly_chart(fig7,use_container_width=True)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

st.subheader("📊 Feature Importance")

importance = model.feature_importances_

feat_df = pd.DataFrame({
    "Feature":X.columns,
    "Importance":importance
}).sort_values(by="Importance")

fig8 = px.bar(
    feat_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="AI Feature Importance"
)

st.plotly_chart(fig8,use_container_width=True)

# =====================================================
# WHAT IF SIMULATION
# =====================================================

st.header("⚡ What-If Simulation")

test_study = st.slider("Change Study Hours",0,12,study_hours)
test_social = st.slider("Change Social Media",0,8,social_media_hours)

test_df = input_df.copy()
test_df["study_hours"] = test_study
test_df["social_media_hours"] = test_social

new_pred = model.predict(test_df)[0]

st.success(
    f"If Study Hours = {test_study} and Social Media = {test_social} → Performance: {new_pred}"
)

# =====================================================
# RECOMMENDATIONS
# =====================================================

st.header("📈 AI Recommendations")

tips=[]

if study_hours <6:
    tips.append("Increase study hours to 6–8 hours.")

if attendance <75:
    tips.append("Improve attendance above 80%.")

if sleep_hours <6:
    tips.append("Sleep at least 6–8 hours.")

if social_media_hours >4:
    tips.append("Reduce social media usage.")

if len(tips)==0:
    tips.append("Great! Your habits are good.")

for tip in tips:
    st.write("✅",tip)

# =====================================================
# DOWNLOAD DATA
# =====================================================

st.header("⬇ Download Dataset")

csv=data.to_csv(index=False)

st.download_button(
    "Download Student Data",
    csv,
    "student_data.csv",
    "text/csv"
)

