import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

with st.sidebar:
    selected = option_menu(
    menu_title="",  # 🔸 No title for cleaner look
    options=["Dashboard", "Dataset"],
    icons=["bar-chart", "table"],
    default_index=0,
    orientation="vertical"
    )


data = pd.read_csv("house_price_dataset.csv")

    
if selected=="Dashboard":
    st.title("Interactive House Price Prediction System")
    st.dataframe(data.head())

    x = data[["Area(sqft)", "Bedrooms", "Age(years)", "LocationScore(1-10)"]]
    y = data["Price(INR)"]

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=50)
    model = LinearRegression()
    model.fit(x_train,y_train)
    if 'section' not in st.session_state:
     st.session_state.section = 'home'

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏠 Predict House Price manully"):
            st.session_state.section = 'home'
    with col2:
        if st.button("📊 Model Performance"):
            st.session_state.section = 'predict'

    # Show selected section
    if st.session_state.section == 'home':
        with st.form("data"):
                area = int(st.number_input("Area (sqft)",value=100,min_value=100))
                bedroom = int(st.number_input("bedrooms",value=1,min_value=1,max_value=10))
                age = int(st.number_input("year",value=1,min_value=1))
                location_score = int(st.number_input("location score", min_value=0, max_value=10))

                input_data = [[area, bedroom, age, location_score]]  # use local variables only
                price = model.predict(input_data)
                submit = st.form_submit_button("pridict price")
                if submit:
                     st.success(f"Predicted Price: ₹ {int(round(price[0])):,}")
                     
       
        
    elif st.session_state.section == 'predict':
        y_pred = model.predict(x_test)
        st.subheader("data pridiction")

        r2 = r2_score(y_test, y_pred)
        st.write("R² Score (model accuracy):", r2)
        print("\n")

        y_pred = np.round(y_pred).astype(int)  # This removes decimals fully

        results = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_pred
        })
        st.table(results)  
        st.header("Model Performance: Actual vs Predicted")
        st.scatter_chart(results)

if selected=="Dataset":
    st.subheader("📊 Dataset")
    st.dataframe(data)