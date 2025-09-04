import streamlit as st
import requests
import folium
from streamlit_folium import st_folium

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="House Price Prediction", layout="wide")

# ==================== STYLING ====================
st.markdown(
    """
    <style>
        .main-title { text-align:center; color:#4B0082; font-size:40px; font-weight:800; }
        .stButton>button {
            background:#4B0082; color:#fff; padding:0.75em 2em; border-radius:12px; border:none;
            font-size:18px; font-weight:700;
        }
        .stButton>button:hover { background:#5a00a3; color:#fff; }
        .prediction-box {
            background:#4B0082; color:#fff; padding:20px; border-radius:12px; text-align:center;
            font-size:22px; font-weight:800;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='main-title'>🏠 House Price Prediction</div>", unsafe_allow_html=True)

# ==================== CONSTANTS ====================
SWAP_LAT_LON_FOR_MODEL = True  # keep True unless retrained with corrected LAT/LON

# India bounds
MIN_LAT, MAX_LAT = 6.0, 38.0
MIN_LON, MAX_LON = 68.0, 98.0

# Default city coordinates
city_coords = {
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Ghaziabad": (28.6692, 77.4538),
    "Jaipur": (26.9124, 75.7873),
    "Kolkata": (22.5726, 88.3639),
    "Lalitpur": (24.6869, 78.4183),
    "Maharashtra": (19.7515, 75.7139),
    "Mumbai": (19.0760, 72.8777),
    "Noida": (28.5355, 77.3910),
    "Other": (22.9734, 78.6569),  # India center
    "Pune": (18.5204, 73.8567),
}

# ==================== INPUTS ====================
st.subheader("🏡 Property Details")
col1, col2, col3 = st.columns(3)

with col1:
    UNDER_CONSTRUCTION = st.selectbox("Under Construction", ["No", "Yes"], index=0)
    RERA = st.selectbox("RERA Registered", ["No", "Yes"], index=1)
    BHK_NO = st.number_input("Number of BHK", min_value=1, max_value=10, value=3)
    SQUARE_FT = st.number_input("Area in Square Feet", min_value=100, max_value=10000, value=1000)

with col2:
    BATHROOM = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
    FURNISHING = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Furnished"])
    AGE = st.slider("Property Age (Years)", 0, 30, 5)
    FLOOR = st.number_input("Floor Number", min_value=0, max_value=50, value=2)

with col3:
    PARKING = st.selectbox("Car Parking", ["No", "Yes"], index=1)
    READY_TO_MOVE = st.selectbox("Ready to Move", ["No", "Yes"], index=1)
    RESALE = st.selectbox("Resale Property", ["No", "Yes"], index=1)
    city = st.selectbox("City", list(city_coords.keys()))
    seller_type = st.selectbox("Seller Type", ["Builder", "Dealer", "Owner"])

# ==================== LOCATION SELECTION ====================
st.subheader("📍 Location (select a city or click on the map)")

# Session state init
if "lat" not in st.session_state or "lon" not in st.session_state:
    st.session_state.lat, st.session_state.lon = city_coords[city]
if "last_city" not in st.session_state:
    st.session_state.last_city = city

# Update coords when city changes
if city != st.session_state.last_city:
    st.session_state.lat, st.session_state.lon = city_coords[city]
    st.session_state.last_city = city

# Clamp to India bounds
st.session_state.lat = float(min(MAX_LAT, max(MIN_LAT, st.session_state.lat)))
st.session_state.lon = float(min(MAX_LON, max(MIN_LON, st.session_state.lon)))

# Build map
m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=10)
folium.Marker([st.session_state.lat, st.session_state.lon], popup=f"{city}").add_to(m)

# Render map and capture clicks
map_click = st_folium(m, width=750, height=520)

if map_click and map_click.get("last_clicked"):
    clicked_lat = float(map_click["last_clicked"]["lat"])
    clicked_lon = float(map_click["last_clicked"]["lng"])
    st.session_state.lat = float(min(MAX_LAT, max(MIN_LAT, clicked_lat)))
    st.session_state.lon = float(min(MAX_LON, max(MIN_LON, clicked_lon)))
    st.rerun()

st.success(f"📍 Final Location: Latitude = {st.session_state.lat:.6f}, Longitude = {st.session_state.lon:.6f}")

# ==================== PREDICT ====================
predict_clicked = st.button("🔮 Predict Price", use_container_width=True)

if predict_clicked:
    if SWAP_LAT_LON_FOR_MODEL:
        model_longitude = st.session_state.lat   # swapped
        model_latitude  = st.session_state.lon   # swapped
    else:
        model_longitude = st.session_state.lon
        model_latitude  = st.session_state.lat

    payload = {
        "UNDER_CONSTRUCTION": 1 if UNDER_CONSTRUCTION == "Yes" else 0,
        "RERA": 1 if RERA == "Yes" else 0,
        "BHK_NO": int(BHK_NO),
        "SQUARE_FT": int(SQUARE_FT),
        "BATHROOM": int(BATHROOM),
        "FURNISHING": FURNISHING,
        "AGE": int(AGE),
        "FLOOR": int(FLOOR),
        "PARKING": 1 if PARKING == "Yes" else 0,
        "READY_TO_MOVE": 1 if READY_TO_MOVE == "Yes" else 0,
        "RESALE": 1 if RESALE == "Yes" else 0,
        "LONGITUDE": model_longitude,
        "LATITUDE": model_latitude,
        "city": city,
        "seller_type": seller_type,
    }

    try:
        resp = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        raw_price = float(data.get("predicted_price_lacs", 0.0))
        safe_price = max(raw_price, 0.25)  # clamp to minimum

        if raw_price < 0:
            st.warning("Model returned a negative value. Displaying a clamped non-negative price. "
                       "Consider retraining with corrected LAT/LON or a log-transformed target.")

        st.markdown(
            f"<div class='prediction-box'>Predicted Price: {safe_price:.2f} Lacs</div>",
            unsafe_allow_html=True,
        )

    except requests.exceptions.RequestException as e:
        st.error(f"Error contacting prediction API: {e}")
