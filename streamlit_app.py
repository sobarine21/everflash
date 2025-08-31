import streamlit as st
import requests
import json # To pretty-print JSON

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Website Audit App",
    page_icon="🔍",
    layout="centered"
)

st.title("🌐 Website Security Audit")
st.markdown("Enter a URL below to get a security audit report.")

# --- API Endpoint ---
API_URL = "https://cyber-p8a5.onrender.com/audit"

# --- User Input ---
user_url = st.text_input(
    "Enter the URL to audit (e.g., https://www.example.com)",
    placeholder="https://www.google.com"
)

# --- Audit Button ---
if st.button("Run Audit"):
    if user_url:
        st.info(f"Auditing: `{user_url}`... Please wait.")
        try:
            # Prepare the data payload
            payload = {"url": user_url}

            # Make the POST request to the API
            response = requests.post(
                API_URL,
                json=payload, # Use json= for automatic Content-Type: application/json
                timeout=30 # Set a timeout for the request (in seconds)
            )

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                st.success("Audit completed successfully!")
                audit_data = response.json()

                # Display the raw JSON response
                st.subheader("Audit Report (Raw JSON)")
                st.json(audit_data) # Streamlit's built-in JSON display

                # --- Optional: Display formatted results ---
                st.subheader("Key Audit Findings")
                if "audit_report" in audit_data and isinstance(audit_data["audit_report"], dict):
                    report = audit_data["audit_report"]

                    st.markdown("---")
                    st.write(f"**Overall Status:** {report.get('status', 'N/A')}")
                    st.write(f"**Description:** {report.get('description', 'No description available.')}")
                    st.write(f"**Timestamp:** {report.get('timestamp', 'N/A')}")
                    st.markdown("---")

                    if "findings" in report and isinstance(report["findings"], list):
                        st.write("**Detailed Findings:**")
                        if report["findings"]:
                            for i, finding in enumerate(report["findings"]):
                                st.markdown(f"**Finding {i+1}:**")
                                st.write(f"- **Severity:** {finding.get('severity', 'N/A')}")
                                st.write(f"- **Issue:** {finding.get('issue', 'N/A')}")
                                st.write(f"- **Description:** {finding.get('description', 'N/A')}")
                                st.write(f"- **Recommendation:** {finding.get('recommendation', 'N/A')}")
                                st.write(f"- **Category:** {finding.get('category', 'N/A')}")
                                st.markdown("---")
                        else:
                            st.info("No specific findings reported for this URL.")
                    else:
                        st.warning("No 'findings' section found in the audit report.")
                else:
                    st.warning("The 'audit_report' structure was not as expected.")

            else:
                st.error(f"Error during audit: API returned status code {response.status_code}")
                try:
                    error_data = response.json()
                    st.json(error_data)
                except json.JSONDecodeError:
                    st.write(response.text) # Display raw text if not JSON
        except requests.exceptions.Timeout:
            st.error("The request timed out. The server might be busy or the URL is taking too long to respond.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Please check your internet connection or try again later.")
        except requests.exceptions.RequestException as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter a URL to run the audit.")

st.markdown("---")
st.markdown("Powered by your custom API.")
