import streamlit as st
import requests
import json

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
            payload = {"url": user_url}
            response = requests.post(API_URL, json=payload, timeout=60) # Increased timeout for potential longer audits

            if response.status_code == 200:
                audit_data = response.json()
                st.success("Audit completed successfully!")

                # --- DEBUGGING STEP: ALWAYS SHOW RAW JSON ---
                st.subheader("Raw API Response (for Debugging)")
                st.json(audit_data)
                # --- END DEBUGGING STEP ---

                st.subheader("Audit Report Summary")
                # Now, attempt to access 'audit_report'
                if "audit_report" in audit_data and isinstance(audit_data["audit_report"], dict):
                    report = audit_data["audit_report"]

                    st.markdown("---")
                    st.write(f"**Overall Status:** <span style='background-color:#FFF3CD; padding: 4px; border-radius: 5px; color:#856404;'>{report.get('status', 'N/A')}</span>", unsafe_allow_html=True)
                    st.write(f"**Description:** {report.get('description', 'No description available.')}")
                    st.write(f"**Timestamp:** {report.get('timestamp', 'N/A')}")
                    st.markdown("---")

                    # Display Severity Counts
                    st.subheader("Severity Breakdown")
                    severity_counts = report.get("severity_counts", {})
                    if severity_counts:
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Critical", severity_counts.get("critical", 0))
                        col2.metric("High", severity_counts.get("high", 0))
                        col3.metric("Medium", severity_counts.get("medium", 0))
                        col4.metric("Low", severity_counts.get("low", 0))
                    else:
                        st.info("No severity counts available.")
                    st.markdown("---")

                    # Display AI Analysis
                    st.subheader("AI Analysis")
                    ai_analysis = report.get("ai_analysis", {})
                    if ai_analysis and ai_analysis.get("status") == "success":
                        st.markdown(ai_analysis.get("analysis", "No AI analysis content available."))
                    elif ai_analysis and ai_analysis.get("status") == "failed":
                        st.warning("AI analysis failed: " + ai_analysis.get("analysis", "Reason unknown."))
                    else:
                        st.info("No AI analysis available for this report.")
                    st.markdown("---")

                    # Display Detailed Findings
                    st.subheader("Detailed Findings")
                    if "findings" in report and isinstance(report["findings"], list):
                        if report["findings"]:
                            for i, finding in enumerate(report["findings"]):
                                st.markdown(f"**Finding {i+1}:**", unsafe_allow_html=True)
                                severity = finding.get('severity', 'N/A')
                                if severity == "High":
                                    st.markdown(f"- **Severity:** <span style='color:red; font-weight:bold;'>{severity}</span>", unsafe_allow_html=True)
                                elif severity == "Medium":
                                    st.markdown(f"- **Severity:** <span style='color:orange; font-weight:bold;'>{severity}</span>", unsafe_allow_html=True)
                                elif severity == "Low":
                                    st.markdown(f"- **Severity:** <span style='color:green;'>{severity}</span>", unsafe_allow_html=True)
                                else:
                                    st.write(f"- **Severity:** {severity}")

                                st.write(f"- **Issue:** {finding.get('issue', 'N/A')}")
                                st.write(f"- **Description:** {finding.get('description', 'N/A')}")
                                st.write(f"- **Recommendation:** {finding.get('recommendation', 'N/A')}")
                                st.markdown("---")
                        else:
                            st.info("No specific findings reported for this URL.")
                    else:
                        st.warning("No 'findings' section found in the audit report.")
                else:
                    st.error("The 'audit_report' structure was not found or was not as expected in the API response.")

            else:
                st.error(f"Error during audit: API returned status code {response.status_code}")
                try:
                    error_data = response.json()
                    st.subheader("Raw Error Response")
                    st.json(error_data)
                except json.JSONDecodeError:
                    st.write(response.text)
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
