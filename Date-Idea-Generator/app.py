import streamlit as st
from streamlit_card import card
from utils import curate_recommendations, invoke_bedrock, date_prompt

st.sidebar.title("Curate Date Experiences With GenerativeAI")

# Supported locations
location = st.sidebar.selectbox(
    'Location',
    ('New York, New York', 'San Fransisco, California')
)

# Distance willing to travel
miles = int(st.sidebar.slider("Distance Willing To Travel (Miles)", 0.0, 0.0, 10.0))
distance = miles * 1609.34 # converting to meters

# Food interests
cuisine = st.sidebar.selectbox(
    'Cuisine',
    ('Thai Food', 'Italian Food', 'Indian Food', 'Chinese Food', 'Japanese Food', 'American Food')
)

# Other interests
secondary_activity = st.sidebar.selectbox(
    'Post Food Activity',
    ('Comedy Show', 'Speakeasy', 'Rooftop Bar')
)

# Additional comments
#additional_input = st.sidebar.text_input("Anything else you'd like to do on your date?")

# Create a submit button
submit_button = st.sidebar.button("Curate Date")
clear_button = st.sidebar.button("Clear Recommendation")

if submit_button:
    st.header("Recommended Date:")

    # generate recommendations
    recommendations = curate_recommendations(location, distance, cuisine, secondary_activity)
    food_recs = recommendations[0]
    activity_recs = recommendations[1]

    input_prompt = date_prompt.format(
        location=location,
        cuisine=cuisine,
        food_recommendations=food_recs,
        activity=secondary_activity,
        activity_recommendations=activity_recs
    )

    output = invoke_bedrock(input_prompt=input_prompt)
    st.write(output)

if clear_button:
    st.experimental_rerun()
