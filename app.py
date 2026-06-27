import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("🌸 Iris Flower Classification App")

# Load and cache data
@st.cache_data
def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    return X, y, iris.target_names

X, y, target_names = load_data()

# Train and cache model
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(X, y)

# Sidebar inputs
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal length (cm)", float(X.iloc[:,0].min()), float(X.iloc[:,0].max()), float(X.iloc[:,0].mean()))
sepal_width  = st.sidebar.slider("Sepal width (cm)",  float(X.iloc[:,1].min()), float(X.iloc[:,1].max()), float(X.iloc[:,1].mean()))
petal_length = st.sidebar.slider("Petal length (cm)", float(X.iloc[:,2].min()), float(X.iloc[:,2].max()), float(X.iloc[:,2].mean()))
petal_width  = st.sidebar.slider("Petal width (cm)",  float(X.iloc[:,3].min()), float(X.iloc[:,3].max()), float(X.iloc[:,3].mean()))

input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=X.columns)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")
    st.success(f"🌼 Species: **{target_names[prediction]}**")

    st.subheader("Prediction Probabilities")
    st.bar_chart(pd.DataFrame(proba, index=target_names, columns=["Probability"]))
else:
    st.info("⬅️ Masukkan fitur dan klik Predict")

