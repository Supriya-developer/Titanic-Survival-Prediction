import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title='Titanic Survival Prediction',
    page_icon='🚢',
    layout='centered',

)

model= joblib.load('titanic_model.pkl')
sc = joblib.load("titanic_scaler.pkl")
columns = joblib.load("titanic_columns.pkl")


st.title('Titanic Survival Prediction🛳️')
st.write(
    "Welcome! Enter passenger details below "
    "to predict whether the passenger survived."
)

st.divider()

st.sidebar.header('Passenger information👤')
##passenger class
pclass = st.sidebar.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

## Gender
sex = st.sidebar.selectbox(
    "Gender",
    ["male", "female"]
)

# Age
age = st.sidebar.slider(
    "Age",
    min_value=1,
    max_value=80,
    value=25
)

# Siblings and Spouses
sibsp = st.sidebar.number_input(
    "Siblings / Spouses",
    min_value=0,
    max_value=8,
    value=0
)

# Parents and Children
parch = st.sidebar.number_input(
    "Parents / Children",
    min_value=0,
    max_value=6,
    value=0
)

# Ticket Fare
fare = st.sidebar.number_input(
    "Ticket Fare",
    min_value=0.0,
    value=32.0
)

# Embarked
embarked = st.sidebar.selectbox(
    "Embarked",
    ["C", "Q", "S"]
)

st.subheader("📋 Passenger Details")

col1, col2 = st.columns(2)

with col1:
    st.metric("Passenger Class", pclass)
    st.metric("Age", age)
    st.metric("Gender", sex.capitalize())

with col2:
    st.metric("Ticket Fare", fare)
    st.metric("Family Members", sibsp + parch)
    st.metric("Embarked", embarked)

st.divider()

##Prediction Button

if st.button("🔍 Predict Survival", use_container_width=True):

    # Create empty input DataFrame
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=columns
    )

    # Prepare input values
    input_values = {
        "Pclass": pclass,
        "Sex": 1 if sex == "male" else 0,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked_Q": 1 if embarked == "Q" else 0,
        "Embarked_S": 1 if embarked == "S" else 0
    }

    # Fill matching columns
    for col, value in input_values.items():
        if col in columns:
            input_data[col] = value

    # Check whether all expected features were filled
    if not all(col in input_values for col in columns):
        st.error(
            "Input features do not match the saved model columns. "
            "Please check your training columns."
        )

    else:
        # Scale input data
        input_scaled = sc.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)

        # Get survival probability
        probability = model.predict_proba(input_scaled)[0][1]


        st.subheader("🎯 Prediction Result")

        if prediction[0] == 1:

            st.success("❤️ Passenger is likely to Survive!")

        else:

            st.error("❌ Passenger is unlikely to Survive.")

        # Show probability
        st.metric(
            "Survival Probability",
            f"{probability * 100:.2f}%"
        )

        # Show probability progress bar
        st.progress(float(probability))

        # Show explanation
        st.info(
            "This prediction is generated using "
            "your trained Logistic Regression model."
        )


        st.divider()

st.caption(
    "Titanic Survival Prediction | "
    "Machine Learning Project ❤️"
) 