import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
from docx import Document
from reportlab.pdfgen import canvas
from PyPDF2 import PdfReader

# Configuración de Tesseract para idioma español
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Ajustar en despliegue

# --- Preprocesamiento Avanzado ---
def preprocess_image_advanced(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(denoised)
    binarized = cv2.adaptiveThreshold(contrast_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    coords = np.column_stack(np.where(binarized > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = binarized.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(binarized, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return deskewed

# OCR en imagen
def extract_text_from_image(image_path):
    processed_img = preprocess_image_advanced(image_path)
    temp_path = "temp_processed.png"
    cv2.imwrite(temp_path, processed_img)
    text = pytesseract.image_to_string(Image.open(temp_path), lang='spa')
    os.remove(temp_path)
    return text

# OCR en PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "
"
    return text

# Guardar en TXT
def save_as_txt(text):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    with open(temp_file.name, 'w', encoding='utf-8') as f:
        f.write(text)
    return temp_file.name

# Guardar en DOC
def save_as_doc(text):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    doc = Document()
    doc.add_paragraph(text)
    doc.save(temp_file.name)
    return temp_file.name

# Guardar en PDF
def save_as_pdf(text):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp_file.name)
    c.setFont("Helvetica", 12)
    y = 800
    for line in text.split('
'):
        c.drawString(50, y, line)
        y -= 15
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = 800
    c.save()
    return temp_file.name

# --- Interfaz Streamlit ---
st.title("Convertidor OCR Español - Versión Web")
st.write("Sube imágenes o PDFs para extraer texto en español y descargarlo en diferentes formatos.")

uploaded_files = st.file_uploader("Selecciona archivos (Imagen/PDF)", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Procesar OCR"):
        extracted_text = ""
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            if uploaded_file.name.lower().endswith('.pdf'):
                extracted_text += extract_text_from_pdf(temp_path) + "
"
            else:
                extracted_text += extract_text_from_image(temp_path) + "
"
        st.success("Texto extraído correctamente.")
        st.text_area("Texto extraído:", extracted_text, height=300)

        # Botones de descarga
        txt_file = save_as_txt(extracted_text)
        doc_file = save_as_doc(extracted_text)
        pdf_file = save_as_pdf(extracted_text)

        st.download_button("Descargar TXT", data=open(txt_file, "rb"), file_name="resultado.txt")
        st.download_button("Descargar DOCX", data=open(doc_file, "rb"), file_name="resultado.docx")
        st.download_button("Descargar PDF", data=open(pdf_file, "rb"), file_name="resultado.pdf")
