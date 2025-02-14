import pytesseract
from pdf2image import convert_from_path
import pandas as pd
import re
import numpy as np
from tabulate import tabulate
import time
from llm import talk_to_llm
import app


start_time = time.perf_counter()
def extract_nutrients_from_pdf(pdf_path):
    """
    Extract food nutrient information from a PDF file using OCR and print results.
    
    Parameters:
    pdf_path (str): Path to the PDF file
    """
    # Convert PDF to images
    try:
        pages = convert_from_path(pdf_path, poppler_path=r'C:\Program Files\poppler-24.08.0\Library\bin')
    except Exception as e:
        raise Exception(f"Error converting PDF to images: {str(e)}")
    
    # Extract text from each page
    full_text = []
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    for page in pages:
        text = pytesseract.image_to_string(page, )
        full_text.append(text)
    
    # Combine all text
    complete_text = ' '.join(full_text)

    return complete_text
    

if __name__ == "__main__":
    start_time_extraction = time.perf_counter()
    data = extract_nutrients_from_pdf('data/Food_Nutrients_info.pdf')
    end_time_extraction = time.perf_counter()


    print(f"Total time for Data Extraction is {end_time_extraction - start_time_extraction} secs")
    
    prompt_1 = f"""from the given data extract the information of all the food names and their nutrients compostion and give in a dictionary that contains a dictionary for each food, also add the amount of food for compostition like 100gras and dont provide additional information like ingredients etc.
        {data}
        """
    
    start_time_llm = time.perf_counter()
    nuetrients_compostition_dict = talk_to_llm(prompt=prompt_1)
    end_time_llm = time.perf_counter()
    print(f"Total time for LLM Calling is {end_time_llm - start_time_llm} secs")
    #print(compostition_dict)

    prompt_2 = f"""from the given data extract the information of {app.food} food their nutrients compostion from the given dictionary.
        {nuetrients_compostition_dict}
        """

    start_time_llm_2 = time.perf_counter()
    extracted_nutrients = talk_to_llm(prompt=prompt_2)
    end_time_llm_2 = time.perf_counter()
    print(f"Total time for LLM Calling is {end_time_llm_2 - start_time_llm_2} secs")
    print(extracted_nutrients)
