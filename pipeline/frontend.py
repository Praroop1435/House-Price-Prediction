import streamlit as st
import pandas as pd
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("house_price_model.h5")

# Encoding maps
city_map = {
    'Chennai': 0, 'Pune': 1, 'Ludhiana': 2, 'Jodhpur': 3, 'Jaipur': 4, 'Durgapur': 5,
    'Coimbatore': 6, 'Bilaspur': 7, 'New Delhi': 8, 'Ranchi': 9, 'Warangal': 10,
    'Bangalore': 11, 'Nagpur': 12, 'Lucknow': 13, 'Silchar': 14, 'Dehradun': 15,
    'Noida': 16, 'Gaya': 17, 'Jamshedpur': 18, 'Ahmedabad': 19, 'Hyderabad': 20,
    'Faridabad': 21, 'Amritsar': 22, 'Kolkata': 23, 'Dwarka': 24, 'Visakhapatnam': 25,
    'Bhopal': 26, 'Indore': 27, 'Haridwar': 28, 'Mysore': 29, 'Patna': 30, 'Raipur': 31,
    'Vijayawada': 32, 'Trivandrum': 33, 'Kochi': 34, 'Surat': 35, 'Gurgaon': 36,
    'Mangalore': 37, 'Cuttack': 38, 'Bhubaneswar': 39, 'Guwahati': 40, 'Mumbai': 41
}

state_map = {
    'Tamil Nadu': 0, 'Maharashtra': 1, 'Punjab': 2, 'Rajasthan': 3, 'West Bengal': 4,
    'Chhattisgarh': 5, 'Delhi': 6, 'Jharkhand': 7, 'Telangana': 8, 'Karnataka': 9,
    'Uttar Pradesh': 10, 'Assam': 11, 'Uttarakhand': 12, 'Bihar': 13, 'Gujarat': 14,
    'Haryana': 15, 'Andhra Pradesh': 16, 'Madhya Pradesh': 17, 'Kerala': 18, 'Odisha': 19
}

property_map = {"Apartment": 0.0, "Independent House": 1.0, "Villa": 2.0}
furnished_map = {"Unfurnished": 0.0, "Semi-furnished": 1.0, "Furnished": 2.0}
transport_map = {"Low": 0.0, "Medium": 1.0, "High": 2.0}
facing_map = {"North": 0, "East": 1, "South": 2, "West": 3}
owner_map = {"Owner": 2, "Builder": 1, "Agent": 0}
yesno_map = {"No": 0.0, "Yes": 1.0}

# Preprocessing wrapper
def preprocess_input(raw_data):
    processed = pd.DataFrame([{
        "City": city_map[raw_data["City"]],
        "State": state_map[raw_data["State"]],
        "Locality": int(raw_data["Locality"].split("_")[1]),
        "Property_Type": property_map[raw_data["Property_Type"]],
        "BHK": raw_data["BHK"],
        "Size_in_SqFt": raw_data["Size_in_SqFt"],
        "Price_per_SqFt": raw_data["Price_per_SqFt"],
        "Year_Built": raw_data["Year_Built"],
        "Furnished_Status": furnished_map[raw_data["Furnished_Status"]],
        "Floor_No": raw_data["Floor_No"],
        "Total_Floors": raw_data["Total_Floors"],
        "Age_of_Property": raw_data["Age_of_Property"],
        "Nearby_Schools": raw_data["Nearby_Schools"],
        "Nearby_Hospitals": raw_data["Nearby_Hospitals"],
        "Public_Transport_Accessibility": transport_map[raw_data["Public_Transport_Accessibility"]],
        "Parking_Space": yesno_map[raw_data["Parking_Space"]],
        "Security": yesno_map[raw_data["Security"]],
        "Facing": facing_map[raw_data["Facing"]],
        "Owner_Type": owner_map[raw_data["Owner_Type"]],
        "Is_Ready_to_Move": 1 if raw_data["Availability_Status"] == "Ready to Move" else 0,
        "Clubhouse": 1 if "Clubhouse" in raw_data["Amenities"] else 0,
        "Garden": 1 if "Garden" in raw_data["Amenities"] else 0,
        "Gym": 1 if "Gym" in raw_data["Amenities"] else 0,
        "Playground": 1 if "Playground" in raw_data["Amenities"] else 0,
        "Pool": 1 if "Pool" in raw_data["Amenities"] else 0,
    }])
    return processed


# ---------------- UI ---------------- #
st.set_page_config(page_title="🏠 Housing Price Prediction", layout="wide")
st.title("🏡 Housing Price Prediction Dashboard")

with st.sidebar:
    st.header("🔧 Input Features")
    state = st.selectbox("State", list(state_map.keys()))
    city = st.selectbox("City", list(city_map.keys()))
    locality = st.text_input("Locality (e.g., Locality_84)", "Locality_84")
    property_type = st.selectbox("Property Type", list(property_map.keys()))
    bhk = st.number_input("BHK", 1, 10, 2)
    size = st.number_input("Size in SqFt", 200, 10000, 1500)
    price_sqft = st.number_input("Price per SqFt (Lakhs)", 0.01, 1.0, 0.12)
    year_built = st.number_input("Year Built", 1950, 2025, 2010)
    furnished = st.selectbox("Furnished Status", list(furnished_map.keys()))
    floor_no = st.number_input("Floor Number", 0, 50, 5)
    total_floors = st.number_input("Total Floors", 1, 100, 10)
    age = st.number_input("Age of Property (Years)", 0, 100, 12)
    schools = st.number_input("Nearby Schools", 0, 20, 8)
    hospitals = st.number_input("Nearby Hospitals", 0, 20, 3)
    transport = st.selectbox("Public Transport Accessibility", list(transport_map.keys()))
    parking = st.radio("Parking Space", ["Yes", "No"])
    security = st.radio("Security", ["Yes", "No"])
    facing = st.selectbox("Facing", list(facing_map.keys()))
    owner_type = st.selectbox("Owner Type", list(owner_map.keys()))
    availability = st.radio("Availability", ["Ready to Move", "Under Construction"])
    amenities = st.multiselect("Amenities", ["Clubhouse", "Garden", "Gym", "Playground", "Pool"])

if st.button("🚀 Predict Price"):
    raw_input = {
        "City": city,
        "State": state,
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Price_per_SqFt": price_sqft,
        "Year_Built": year_built,
        "Furnished_Status": furnished,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Age_of_Property": age,
        "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Security": security,
        "Facing": facing,
        "Owner_Type": owner_type,
        "Availability_Status": availability,
        "Amenities": amenities
    }

    processed = preprocess_input(raw_input)
    prediction = model.predict(processed)[0][0]

    # ---- Results Section ---- #
    st.subheader("📊 Results")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(" Predicted Price", f"₹ {prediction:,.2f} Lakhs")
