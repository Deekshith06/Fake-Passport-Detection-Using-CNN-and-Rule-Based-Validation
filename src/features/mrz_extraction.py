"""
MRZ EXTRACTION MODULE
=====================
Extract MRZ (Machine Readable Zone) text from passport images using OCR.

Simple implementation for student projects.
"""

import cv2
import numpy as np
import pytesseract
import re


def extract_mrz_from_image(image):
    """
    Extract MRZ text from passport image using OCR.
    
    Steps:
    1. Extract bottom region (where MRZ is located)
    2. Preprocess image for better OCR
    3. Run OCR multiple times with different configs
    4. Validate and clean extracted text
    
    Args:
        image: Passport image (RGB numpy array)
        
    Returns:
        str: Extracted MRZ text (or None if extraction failed)
    """
    
    print("\n" + "="*60)
    print("EXTRACTING MRZ TEXT")
    print("="*60)
    
    try:
        # Step 1: Extract MRZ region (bottom 15% of image)
        h, w = image.shape[:2]
        mrz_region = image[int(h*0.85):, :]
        
        # Step 2: Preprocess for OCR
        # Convert to grayscale
        gray = cv2.cvtColor(mrz_region, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Resize for better OCR (make it bigger)
        scale = 2
        binary = cv2.resize(binary, (binary.shape[1]*scale, binary.shape[0]*scale))
        
        print("✓ Image preprocessed")
        
        # Step 3: Try OCR with different configurations
        mrz_text = None
        
        # Attempt 1: Standard config
        print("\nAttempt 1: Standard OCR...")
        text1 = pytesseract.image_to_string(
            binary,
            config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
        )
        text1 = clean_mrz_text(text1)
        print(f"  Result: {text1[:50] if text1 else 'None'}...")
        
        # Attempt 2: Single line mode
        print("\nAttempt 2: Single line mode...")
        text2 = pytesseract.image_to_string(
            binary,
            config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
        )
        text2 = clean_mrz_text(text2)
        print(f"  Result: {text2[:50] if text2 else 'None'}...")
        
        # Attempt 3: Different preprocessing
        print("\nAttempt 3: Inverted image...")
        inverted = cv2.bitwise_not(binary)
        text3 = pytesseract.image_to_string(
            inverted,
            config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
        )
        text3 = clean_mrz_text(text3)
        print(f"  Result: {text3[:50] if text3 else 'None'}...")
        
        # Step 4: Choose best result
        candidates = [text1, text2, text3]
        candidates = [t for t in candidates if t and len(t) > 20]
        
        if candidates:
            # Choose longest valid-looking result
            mrz_text = max(candidates, key=len)
            print(f"\n✓ Best MRZ extracted (length: {len(mrz_text)})")
        else:
            print("\n✗ No valid MRZ text extracted")
            
        print("="*60)
        return mrz_text
        
    except Exception as e:
        print(f"\n✗ MRZ extraction error: {e}")
        print("="*60)
        return None


def clean_mrz_text(text):
    """
    Clean and validate extracted MRZ text.
    
    Rules:
    - Remove spaces and newlines
    - Convert to uppercase
    - Replace similar characters (O→0, I→1, etc.)
    - Keep only valid MRZ characters
    """
    
    if not text:
        return None
    
    # Remove whitespace
    text = text.replace(' ', '').replace('\n', '').replace('\r', '')
    
    # Convert to uppercase
    text = text.upper()
    
    # Replace common OCR mistakes
    text = text.replace('O', '0')  # O to 0
    text = text.replace('I', '1')  # I to 1
    text = text.replace('|', '1')  # | to 1
    text = text.replace('S', '5')  # S to 5 (sometimes)
    
    # Keep only valid MRZ characters
    text = re.sub(r'[^A-Z0-9<]', '', text)
    
    # MRZ should be at least 20 characters
    if len(text) < 20:
        return None
    
    return text


def validate_mrz_format(mrz_text):
    """
    Check if extracted text looks like valid MRZ format.
    
    Returns:
        tuple: (is_valid, confidence_score)
    """
    
    if not mrz_text:
        return False, 0.0
    
    score = 0.0
    
    # Check length (MRZ lines are usually 44 chars)
    if 30 <= len(mrz_text) <= 90:
        score += 0.3
    
    # Check for '<' characters (fillers)
    filler_count = mrz_text.count('<')
    if filler_count > 0:
        score += 0.3
    
    # Check for mix of letters and numbers
    has_letters = any(c.isalpha() for c in mrz_text)
    has_numbers = any(c.isdigit() for c in mrz_text)
    if has_letters and has_numbers:
        score += 0.4
    
    is_valid = score >= 0.6
    
    return is_valid, score
