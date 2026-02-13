import streamlit as st
import pandas as pd
import joblib

def display_map(options):
    return {opt.title(): opt for opt in options}


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Indian House Price Predictor",
    page_icon="🏡",
    layout="centered"
)

# --------------------------------------------------
# Load trained PIPELINE model
# --------------------------------------------------
model = joblib.load("model_xgb_pipeline.joblib")

st.title("🏡 Indian House Price Predictor")
st.write("Predict house prices using location, amenities, and property details.")

st.divider()

# --------------------------------------------------
# Input options
# --------------------------------------------------
LOCATIONS = [
    'thane','navi-mumbai','nagpur','mumbai','ahmedabad','bangalore',
    'chennai','gurgaon','hyderabad','indore','jaipur','kolkata',
    'lucknow','new-delhi','noida','pune','agra','ahmadnagar',
    'allahabad','aurangabad','badlapur','belgaum','bhiwadi',
    'bhiwandi','bhopal','bhubaneswar','chandigarh','coimbatore',
    'dehradun','durgapur','ernakulam','faridabad','ghaziabad',
    'goa','greater-noida','guntur','guwahati','gwalior','haridwar',
    'jabalpur','jamshedpur','jodhpur','kalyan','kanpur','kochi',
    'kozhikode','ludhiana','madurai','mangalore','mohali','mysore',
    'nashik','navsari','nellore','palakkad','palghar','panchkula',
    'patna','pondicherry','raipur','rajahmundry','ranchi','satara',
    'shimla','siliguri','solapur','sonipat','surat','thrissur',
    'tirupati','trichy','trivandrum','udaipur','udupi','vadodara',
    'vapi','varanasi','vijayawada','visakhapatnam','vrindavan',
    'zirakpur'
]

FACING = [
    'East','West','North','North - East',
    'North - West','South','South - East','South -West'
]

OWNERSHIP = [
    'Freehold','Co-operative Society',
    'Power Of Attorney','Leasehold'
]

FURNISHING = [
    'Unfurnished','Semi-Furnished','Furnished'
]

TRANSACTION = [
    'Resale','New Property','Other','Rent/Lease'
]

YES_NO = ['Yes', 'No']

LOCATION_MAP = display_map(LOCATIONS)
FACING_MAP = display_map(FACING)
OWNERSHIP_MAP = display_map(OWNERSHIP)
FURNISHING_MAP = display_map(FURNISHING)
TRANSACTION_MAP = display_map(TRANSACTION)
YES_NO_MAP = display_map(YES_NO)

# --------------------------------------------------
# Input layout
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    location_display = st.selectbox("Location", LOCATION_MAP.keys())
    location = LOCATION_MAP[location_display]
    facing_display = st.selectbox("Facing", FACING_MAP.keys())
    facing = FACING_MAP[facing_display]
    ownership_display = st.selectbox("Ownership Type", OWNERSHIP_MAP.keys())
    ownership = OWNERSHIP_MAP[ownership_display]
    furnishing_display = st.selectbox("Furnishing Status", FURNISHING_MAP.keys())
    furnishing = FURNISHING_MAP[furnishing_display]
    transaction_display = st.selectbox("Transaction Type", TRANSACTION_MAP.keys())
    transaction = TRANSACTION_MAP[transaction_display]
    bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=3)

with col2:
    carpet_area = st.number_input("Carpet Area (sq ft)", min_value=100, max_value=10000, value=1000)
    
    bathroom = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    balcony = st.number_input("Balcony", min_value=0, max_value=10, value=1)
    floor = st.number_input("Floor", min_value=0, max_value=50, value=1)
    main_road_display = st.selectbox("Main Road Access", YES_NO_MAP.keys())
    main_road = YES_NO_MAP[main_road_display]
    garden_park_display = st.selectbox("Garden / Park", YES_NO_MAP.keys())
    garden_park = YES_NO_MAP[garden_park_display]


st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔮 Predict Price"):
    with st.spinner("Predicting price..."):
        try:
            input_df = pd.DataFrame({
                "Location": [location],
                "Facing": [facing],
                "Ownership": [ownership],
                "Furnishing": [furnishing],
                "Transaction": [transaction],
                "Main Road": [main_road],
                "Garden/Park": [garden_park],
                "Carpet Area": [int(carpet_area)],
                "Bathroom": [int(bathroom)],
                "Balcony": [int(balcony)],
                "Rooms": [int(bhk)],
                "Floor": [int(floor)]
            })

            prediction = model.predict(input_df)

            st.success(f"💰 Estimated Price: ₹ {prediction[0]:,.2f}")

        except Exception as e:
            st.error("Prediction failed. Please check inputs.")
            st.exception(e)
