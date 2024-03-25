import streamlit as st
import tempfile
import fitz
# import pytesseract
from PIL import Image
import google.generativeai as genai
from pathlib import Path
import pandas as pd

# Configure Google API Key
genai.configure(api_key="AIzaSyBtkn51M4i35dzxEQyRJl8dBE1JcDRourg")

# Set up the model
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-pro-vision",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

def input_image_setup(uploaded_file):
    if not uploaded_file:
        raise FileNotFoundError("Please upload an image to ask questions.")

    temp_path = Path("temp_image.jpg")
    temp_path.write_bytes(uploaded_file.read())

    image_data = temp_path.read_bytes()

    return image_data, temp_path

def generate_gemini_response(image_data, question_prompt):
    response = model.generate_content([{"mime_type": "image/jpeg", "data": image_data}, question_prompt])
    return response.text

def read_pdf_and_generate_response(pdf_data, question_prompt):
    images = convert_pdf_to_images(pdf_data)
    response_texts = []
    for image_data in images:
        response_texts.append(generate_gemini_response(image_data, question_prompt))
    return response_texts

def convert_pdf_to_images(pdf_data):
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
        temp.write(pdf_data)
        temp_path = temp.name

    images = []
    with tempfile.TemporaryDirectory() as temp_dir:
        doc = fitz.open(temp_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=170)
            img_temp_path = Path(temp_dir) / f"page_{page_num}.jpg"
            pix.save(str(img_temp_path), jpg_quality=98)
            img_data = img_temp_path.read_bytes()
            images.append(img_data)

    return images

input_prompt = """
               You are an expert in understanding invoices and receipt.
               You will receive input images as invoices &
               you will have to answer questions based on the input image
               """

def main():
    st.title("AI Invoice Scanner v0.1")

    uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
    question_prompt = st.text_input("Question Prompt", "")

    if st.button("Generate Responses"):
        response_data = []
        for uploaded_file in uploaded_files:
            try:
                response_texts = read_pdf_and_generate_response(uploaded_file.read(), question_prompt)
                response_data.append({"PDF File": uploaded_file.name, "Responses": response_texts})
            except FileNotFoundError as e:
                st.warning(str(e))
        
        if response_data:
            df = pd.DataFrame(response_data)
            st.subheader("Generated Responses:")
            st.write(df)

if __name__ == "__main__":
    main()
