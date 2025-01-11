import streamlit as st
import PyPDF2
import google.generativeai as genai

# Configure the API key securely from Streamlit's secrets
# Make sure to add GOOGLE_API_KEY in secrets.toml (for local) or Streamlit Cloud Secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to generate flashcards using Gemini API
def generate_flashcards_with_gemini(text):
    try:
        # Load and configure the Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate flashcards from the provided text
        prompt = f"Generate study flashcards based on the following text:\n\n{text}"
        response = model.generate_content(prompt)
        
        # Return the generated flashcards
        return response.text
    except Exception as e:
        st.error(f"Error generating flashcards: {e}")
        return ""

# Streamlit interface
st.title("AI Flashcard Generator with Gemini API")

# File uploader for PDF or text files
uploaded_file = st.file_uploader("Upload a PDF or Text file", type=["pdf", "txt"])

if uploaded_file is not None:
    # Extract text based on file type
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        text = uploaded_file.getvalue().decode("utf-8")
    
    # Display the extracted text (optional)
    st.subheader("Extracted Text:")
    st.text_area("Text Preview", text, height=200)

    # Generate flashcards
    if st.button("Generate Flashcards"):
        flashcards = generate_flashcards_with_gemini(text)
        if flashcards:
            st.subheader("Generated Flashcards:")
            st.text_area("Flashcards", flashcards, height=300)
