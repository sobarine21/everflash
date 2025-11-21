import streamlit as st
import requests
import json

st.set_page_config(page_title="Supabase Internal DB Query", layout="wide")

st.title("🔍 Supabase Internal DB Query Runner")
st.write("Run SQL on your Supabase DB using your Edge Function.")

# -------------------------
# Load secrets
# -------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]        # example: https://xxxx.supabase.co
FUNCTION_PATH = "/functions/v1/internal-db-query"

SERVICE_KEY = st.secrets["SUPABASE_SERVICE_ROLE"]   # keep service-role key in secrets!

ENDPOINT = SUPABASE_URL + FUNCTION_PATH

# -------------------------
# User Input UI
# -------------------------
st.subheader("Query Inputs")

database_id = st.text_input("Database ID", placeholder="your-database-id")
query_text = st.text_area("SQL Query", value="SELECT * FROM users LIMIT 10")

run_btn = st.button("Run Query")

# -------------------------
# Execute
# -------------------------
if run_btn:
    if not database_id or not query_text:
        st.error("Database ID and SQL Query are required.")
        st.stop()

    payload = {
        "database_id": database_id,
        "query_text": query_text
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": SERVICE_KEY
    }

    st.info("Running query...")

    try:
        response = requests.post(ENDPOINT, headers=headers, data=json.dumps(payload))

        if response.status_code != 200:
            st.error(f"❌ Error: {response.status_code}")
            st.code(response.text)
        else:
            result = response.json()

            st.success("Query executed successfully!")

            # Show JSON response
            st.subheader("Raw JSON Response")
            st.json(result)

            # Try showing as a dataframe
            if isinstance(result, dict) and "rows" in result:
                st.subheader("Table Output")
                st.dataframe(result["rows"])
            else:
                st.info("No tabular data found in response.")

    except Exception as e:
        st.error("Request failed.")
        st.exception(e)
