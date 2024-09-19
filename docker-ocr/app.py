from flask import Flask, request, jsonify, abort, make_response
import cv2
import pytesseract
from pdf2image import convert_from_path
import os
import tempfile

app = Flask(__name__)

def extract_text_from_file(file, language='spa'):
    """
    Extrae texto de un archivo (PDF, JPG, PNG) y aplica optimización.

    Args:
        file: Objeto de archivo subido.
        language (str, optional): Idioma del texto (por defecto 'spa').
    Returns:
        str: Texto extraído.
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        file.save(temp_file.name)
        file_path = temp_file.name

        if file_path.endswith('.pdf'):
            print(f"Es un PDF: {file_path}")
            text = extract_text_from_pdf_with_ocr(file_path, language=language)
        elif file_path.endswith(('.jpg', '.jpeg', '.png')):
            print(f"Es una imagen: {file_path}")
            img = cv2.imread(file_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            config = f'--psm 6 -l {language}'
            text = pytesseract.image_to_string(thresh, config=config)
        else:
            print(f"No se reconoce el archivo: {file_path}")
            text = ""
        os.remove(file_path)
    return text

def extract_text_from_pdf_with_ocr(pdf_path, language, dpi=200):
    pages = convert_from_path(pdf_path, dpi=dpi)
    extracted_text = ""
    for page_num, page in enumerate(pages):
        print(f"Procesando página {page_num + 1} de {len(pages)}...")
        # Incluimos el parámetro de lenguaje en la configuración de pytesseract
        config = f'-l {language}'
        extracted_text += pytesseract.image_to_string(page, config=config)
    return extracted_text

@app.route('/ocr', methods=['POST'])
def ocr():
    """
    Compara el texto extraído de un archivo con una cadena de entrada.

    Returns:
        JSON: Resultado de la comparación (porcentaje de coincidencia, palabras encontradas, umbral superado).
    """

    # Validación de campos obligatorios
    if 'file' not in request.files:
        error_message = "El campo 'file' es obligatorio."
        return make_response(jsonify({'status': 400, 'error': error_message}), 400)

    file = request.files['file']
    if not file.filename.endswith(('.pdf', '.jpg', '.jpeg', '.png')):
        error_message = "El archivo debe ser de tipo PDF, JPG, JPEG o PNG."
        return make_response(jsonify({'status': 400, 'error': error_message}), 400)

    if file.content_length > 5 * 1024 * 1024:  # 5MB
        error_message = "El archivo no debe superar los 5MB de tamaño."
        return make_response(jsonify({'status': 400, 'error': error_message}), 400)

    if 'search_string' not in request.form:
        error_message = "El campo 'search_string' es obligatorio."
        return make_response(jsonify({'status': 400, 'error': error_message}), 400)

    search_string = request.form['search_string']
    if len(search_string) > 255:
        error_message = "La cadena de búsqueda no debe superar los 255 caracteres."
        return make_response(jsonify({'status': 400, 'error': error_message}), 400)


    file = request.files['file']
    search_string = request.form['search_string']
    threshold = float(request.form.get('threshold', 0.5))
    language = request.form.get('language', 'spa') # Parámetro opcional de lenguaje

    extracted_text = extract_text_from_file(file, language=language)
    extracted_words = set(extracted_text.lower().split())
    search_words = set(search_string.lower().split())

    matched_words = extracted_words.intersection(search_words)
    percentage = len(matched_words) / len(search_words) if len(search_words) > 0 else 0

    result = {
        'percentage': percentage,
        'matched_words': list(matched_words),
        'threshold_exceeded': percentage >= threshold
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)