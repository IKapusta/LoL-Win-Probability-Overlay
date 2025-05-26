import easyocr

def extract_numbers_with_easyocr(image_path, allowList ='0123456789',use_gpu=True):
    """
    Extract text from an image using EasyOCR.

    Parameters:
    - image_path: Path to the input image.
    - use_gpu: Boolean indicating whether to use GPU for processing (default is True).

    Returns:
    - Extracted text as a list of strings 
    """
    reader = easyocr.Reader(['en'], gpu=use_gpu)
 
    results = reader.readtext(image_path, detail=0, allowlist=allowList)
    
    return results[0]

# Example usage
if __name__ == "__main__":
    image_path = "./scoreboards/upscaled_area_0_1.png"  
    extracted_text = extract_numbers_with_easyocr(image_path, use_gpu=True)
    
    print("Extracted Numbers:")
    for text in extracted_text:
        print(text)
