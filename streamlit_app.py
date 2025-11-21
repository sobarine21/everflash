import streamlit as st
import requests
import pandas as pd
import json

st.set_page_config(page_title="Supabase DB Query", layout="wide")

st.title("🔍 Supabase Internal DB Query Tool")

# ---------------------------------------------------------
# Load from st.secrets
# ---------------------------------------------------------
API_URL = st.secrets["supabase"]["db_function_url"]
API_KEY = st.secrets["supabase"]["api_key"]
DATABASE_ID = st.secrets["supabase"]["database_id"]

# ---------------------------------------------------------
# User SQL input
# ---------------------------------------------------------
st.subheader("Enter SQL Query")
query_text = st.text_area(
    "Write your SQL here",
    value="SELECT 1 as test_value;",
    height=150
)

if st.button("Run Query"):
    if not query_text.strip():
        st.error("Query cannot be empty.")
        st.stop()

    payload = {
        "database_id": DATABASE_ID,
        "query_text": query_text
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }

    with st.spinner("Running query..."):
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # ---------------------------------------------------------
            # Display Result
            # ---------------------------------------------------------
            st.success("Query executed successfully!")

            if isinstance(data, dict) and "data" in data:
                df = pd.DataFrame(data["data"])
                st.dataframe(df, use_container_width=True)
            else:
                st.write(data)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.code(response.text)
