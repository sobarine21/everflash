import streamlit as st
import PyPDF2
import google.generativeai as genai
from fpdf import FPDF
import os

# Configure the API key securely from Streamlit's secrets
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

# Function to create a PDF of flashcards
def create_flashcards_pdf(flashcards, filename="flashcards.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add the flashcards content to the PDF
    flashcard_list = flashcards.split('\n\n')
    for flashcard in flashcard_list:
        question_answer = flashcard.split("\n")
        if len(question_answer) == 2:
            question, answer = question_answer
            pdf.multi_cell(0, 10, f"Question: {question}", align='L')
            pdf.multi_cell(0, 10, f"Answer: {answer}\n", align='L')
        else:
            pdf.multi_cell(0, 10, f"{flashcard}\n\n", align='L')
    
    # Save the PDF to a file
    pdf.output(filename)

# Streamlit interface
st.title("AI Flashcard Generator with Gemini API")
st.write("Upload a PDF or text file to generate flashcards based on the content.")

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
            flashcard_list = flashcards.split('\n\n')
            
            # Display flashcards on the page
            for flashcard in flashcard_list:
                st.markdown(f"**Flashcard:** {flashcard}")
            
            # Create and provide downloadable PDF
            pdf_filename = "generated_flashcards.pdf"
            create_flashcards_pdf(flashcards, filename=pdf_filename)

            st.download_button(
                label="Download Flashcards as PDF",
                data=open(pdf_filename, "rb").read(),
                file_name=pdf_filename,
                mime="application/pdf"
            )
            
            # Clean up the generated PDF file
            os.remove(pdf_filename)
