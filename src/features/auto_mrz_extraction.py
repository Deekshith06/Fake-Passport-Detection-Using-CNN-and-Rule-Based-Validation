"""
AGGRESSIVE MRZ EXTRACTION - MAXIMUM AUTOMATIC SUCCESS
======================================================
Tries every technique to extract MRZ automatically.
Validation happens later via checksums - safe to be aggressive here.
"""

import cv2
import numpy as np
import re
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


def enhance_image(img):
    """Aggressive image enhancement for better OCR."""
    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(sharpened, None, 10, 7, 21)
    
    return denoised


def preprocess_aggressive_1(mrz_region):
    """Method 1: CLAHE + Adaptive threshold."""
    gray = cv2.cvtColor(mrz_region, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_image(gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(enhanced)
    blur = cv2.medianBlur(contrast, 3)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh


def preprocess_aggressive_2(mrz_region):
    """Method 2: Strong contrast + Otsu."""
    gray = cv2.cvtColor(mrz_region, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_image(gray)
    contrast = cv2.convertScaleAbs(enhanced, alpha=2.0, beta=20)
    blur = cv2.GaussianBlur(contrast, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def preprocess_aggressive_3(mrz_region):
    """Method 3: Morphological operations."""
    gray = cv2.cvtColor(mrz_region, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_image(gray)
    _, thresh = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morph


def preprocess_aggressive_4(mrz_region):
    """Method 4: Edge enhancement."""
    gray = cv2.cvtColor(mrz_region, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_image(gray)
    edges = cv2.Canny(enhanced, 50, 150)
    dilated = cv2.dilate(edges, None, iterations=1)
    return 255 - dilated


def preprocess_aggressive_5(mrz_region):
    """Method 5: Simple high contrast."""
    gray = cv2.cvtColor(mrz_region, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_image(gray)
    contrast = cv2.convertScaleAbs(enhanced, alpha=2.5, beta=0)
    return contrast


def auto_correct_mrz_characters(text):
    """Aggressive character correction."""
    replacements = {
        'O': '0', 'Q': '0', 'D': '0',
        'I': '1', 'l': '1', '|': '1',
        'Z': '2', 'S': '5', 'B': '8',
        'G': '6', 'T': '7',
        ' ': '<', '.': '<', ',': '<', '*': '<',
        '@': 'A', '&': 'A',
    }
    
    result = text.upper()
    for old, new in replacements.items():
        result = result.replace(old, new)
    
    # Remove completely invalid characters
    result = re.sub(r'[^A-Z0-9<\n]', '', result)
    
    return result


def extract_best_lines(text, target_count=2):
    """Extract most MRZ-like lines from OCR output."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    # Score each line by how MRZ-like it is
    scored_lines = []
    for line in lines:
        score = 0
        
        # Length near 44
        if 35 <= len(line) <= 50:
            score += 10
            if 42 <= len(line) <= 46:
                score += 20
        
        # Has < characters
        if '<' in line:
            score += line.count('<') * 2
        
        # Has uppercase letters
        score += sum(1 for c in line if c.isupper()) * 0.5
        
        # Has digits
        score += sum(1 for c in line if c.isdigit()) * 0.5
        
        # Starts with P< or I<
        if line.startswith('P<') or line.startswith('I<'):
            score += 50
        
        scored_lines.append((score, line))
    
    # Sort by score
    scored_lines.sort(reverse=True, key=lambda x: x[0])
    
    # Return top N lines
    return [line for score, line in scored_lines[:target_count]]


def fix_line_length(line, target_length=44):
    """Pad or trim line to target length."""
    if len(line) < target_length:
        # Pad with <
        return line + '<' * (target_length - len(line))
    elif len(line) > target_length:
        # Trim
        return line[:target_length]
    return line


def extract_mrz_aggressive(image):
    """
    AGGRESSIVE multi-pass extraction - tries everything.
    
    Returns best extraction attempt, even if not perfect.
    Checksum validation will catch errors downstream.
    """
    if not TESSERACT_AVAILABLE:
        return {
            'mrz_text': None,
            'confidence': 0.0,
            'status': 'FAILED',
            'method': 'N/A',
            'message': 'Tesseract not available - please install'
        }
    
    try:
        # Extract MRZ region (bottom 25-30%)
        h, w = image.shape[:2]
        
        # Try multiple crop positions
        crops = [
            image[int(h * 0.70):h, :],  # Bottom 30%
            image[int(h * 0.72):h, :],  # Bottom 28%
            image[int(h * 0.75):h, :],  # Bottom 25%
        ]
        
        # All preprocessing methods
        methods = [
            ('CLAHE', preprocess_aggressive_1),
            ('Contrast', preprocess_aggressive_2),
            ('Morph', preprocess_aggressive_3),
            ('Edge', preprocess_aggressive_4),
            ('Simple', preprocess_aggressive_5),
        ]
        
        # OCR configurations to try
        configs = [
            "--psm 6 --oem 3",  # Standard
            "--psm 4 --oem 3",  # Single column
            "--psm 11 --oem 3",  # Sparse text
        ]
        
        all_results = []
        
        # Try every combination
        for crop in crops:
            for method_name, preprocess_func in methods:
                for config in configs:
                    try:
                        processed = preprocess_func(crop)
                        raw_text = pytesseract.image_to_string(processed, config=config)
                        corrected_text = auto_correct_mrz_characters(raw_text)
                        lines = extract_best_lines(corrected_text, 2)
                        
                        if len(lines) >= 2:
                            # Fix lengths
                            line1 = fix_line_length(lines[0])
                            line2 = fix_line_length(lines[1])
                            
                            # Calculate simple confidence
                            score = 0
                            if line1.startswith('P<') or line1.startswith('I<'):
                                score += 30
                            if len(line1) == 44 and len(line2) == 44:
                                score += 30
                            if line1.count('<') + line2.count('<') >= 10:
                                score += 20
                            
                            valid_chars = sum(1 for c in (line1 + line2) if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<')
                            char_ratio = valid_chars / 88.0
                            score += char_ratio * 20
                            
                            confidence = min(score / 100.0, 1.0)
                            
                            all_results.append({
                                'text': f"{line1}\n{line2}",
                                'confidence': confidence,
                                'method': method_name
                            })
                    except:
                        continue
        
        # Pick best result
        if all_results:
            best = max(all_results, key=lambda x: x['confidence'])
            
            return {
                'mrz_text': best['text'],
                'confidence': best['confidence'],
                'status': 'SUCCESS' if best['confidence'] > 0.6 else 'LOW_CONFIDENCE',
                'method': best['method'],
                'message': f"MRZ extracted (Confidence: {best['confidence']:.0%}) - Verify if needed"
            }
        
        # Nothing worked - return empty but not None
        return {
            'mrz_text': '',
            'confidence': 0.0,
            'status': 'FAILED',
            'method': 'All methods failed',
            'message': 'Could not extract MRZ - Text box is empty, please enter manually'
        }
    
    except Exception as e:
        return {
            'mrz_text': '',
            'confidence': 0.0,
            'status': 'FAILED',
            'method': 'Error',
            'message': 'Extraction error - Please enter MRZ manually'
        }


def extract_mrz_with_fallback(image):
    """Main entry point."""
    return extract_mrz_aggressive(image)


if __name__ == "__main__":
    if TESSERACT_AVAILABLE:
        print("✅ Aggressive MRZ extraction ready")
    else:
        print("❌ Install: pip install pytesseract")
