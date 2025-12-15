"""
ADVANCED MRZ EXTRACTION MODULE
==============================
High-accuracy MRZ extraction using multiple OCR engines and advanced preprocessing.

Uses best available OCR engines for maximum accuracy:
- Tesseract (traditional, good baseline)
- EasyOCR (deep learning, better for low quality)
- Ensemble voting for best results
"""

import cv2
import numpy as np
import re

# Try to import multiple OCR engines
TESSERACT_AVAILABLE = False
EASYOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    pass

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    pass


class AdvancedMRZExtractor:
    """High-accuracy MRZ extraction using best available methods."""
    
    def __init__(self):
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                # Initialize EasyOCR (may download models on first use)
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
            except:
                pass
    
    def preprocess_for_mrz(self, image):
        """
        Advanced preprocessing specifically optimized for MRZ text.
        Returns multiple preprocessed versions for ensemble extraction.
        """
        h, w = image.shape[:2]
        mrz_region = image[int(h * 0.72):h, :]
        
        preprocessed_images = []
        
        # Method 1: High contrast with sharpening
        gray = cv2.cvtColor(mrz_region, cv2.COLOR_BGR2GRAY)
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(sharpened)
        _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(('CLAHE+Sharpen', thresh1))
        
        # Method 2: Adaptive threshold
        blur = cv2.medianBlur(enhanced, 3)
        adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        preprocessed_images.append(('Adaptive', adaptive))
        
        # Method 3: Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        preprocessed_images.append(('Morphology', morph))
        
        # Method 4: Denoised version
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        _, thresh2 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(('Denoised', thresh2))
        
        return preprocessed_images
    
    def extract_with_tesseract(self, processed_images):
        """Extract MRZ using Tesseract OCR."""
        if not TESSERACT_AVAILABLE:
            return []
        
        results = []
        configs = [
            "--psm 6 --oem 3",
            "--psm 4 --oem 3", 
            "--psm 11 --oem 3",
        ]
        
        for method_name, img in processed_images:
            for config in configs:
                try:
                    text = pytesseract.image_to_string(img, config=config)
                    mrz_lines = self.parse_mrz_from_text(text)
                    if mrz_lines:
                        results.append({
                            'engine': f'Tesseract ({method_name})',
                            'line1': mrz_lines[0],
                            'line2': mrz_lines[1],
                            'confidence': 0.7
                        })
                except:
                    continue
        
        return results
    
    def extract_with_easyocr(self, processed_images):
        """Extract MRZ using EasyOCR (deep learning)."""
        if not self.easyocr_reader:
            return []
        
        results = []
        for method_name, img in processed_images:
            try:
                # EasyOCR works on grayscale or BGR
                result = self.easyocr_reader.readtext(img, detail=1)
                
                # Combine detected text
                text_lines = [text for (bbox, text, prob) in result if prob > 0.3]
                full_text = '\n'.join(text_lines)
                
                mrz_lines = self.parse_mrz_from_text(full_text)
                if mrz_lines:
                    avg_confidence = sum([prob for (_, _, prob) in result]) / len(result) if result else 0
                    results.append({
                        'engine': f'EasyOCR ({method_name})',
                        'line1': mrz_lines[0],
                        'line2': mrz_lines[1],
                        'confidence': avg_confidence
                    })
            except:
                continue
        
        return results
    
    def parse_mrz_from_text(self, text):
        """Extract and clean MRZ lines from OCR output."""
        # Clean text
        text = text.upper()
        text = re.sub(r'[^A-Z0-9<\n]', '', text)
        
        # Find lines that look like MRZ
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        candidates = []
        
        for line in lines:
            if 35 <= len(line) <= 50:
                # Fix length
                if len(line) < 44:
                    line = line + '<' * (44 - len(line))
                elif len(line) > 44:
                    line = line[:44]
                
                # Check if it looks like MRZ
                if line.startswith('P<') or line.startswith('I<') or line[0].isalnum():
                    candidates.append(line)
        
        # Need exactly 2 lines
        if len(candidates) >= 2:
            return [candidates[0], candidates[1]]
        
        return None
    
    def ensemble_extract(self, image):
        """
        Use ensemble of all available OCR engines for best accuracy.
        
        Returns:
            dict with best extraction result
        """
        # Preprocess
        preprocessed = self.preprocess_for_mrz(image)
        
        # Extract with all engines
        all_results = []
        
        if TESSERACT_AVAILABLE:
            all_results.extend(self.extract_with_tesseract(preprocessed))
        
        if self.easyocr_reader:
            all_results.extend(self.extract_with_easyocr(preprocessed))
        
        if not all_results:
            return {
                'success': False,
                'message': 'No OCR engine available. Install: pip install pytesseract easyocr',
                'mrz_text': None
            }
        
        # Vote / pick best result
        best = max(all_results, key=lambda x: x['confidence'])
        
        return {
            'success': True,
            'mrz_text': f"{best['line1']}\n{best['line2']}",
            'line1': best['line1'],
            'line2': best['line2'],
            'engine': best['engine'],
            'confidence': best['confidence'],
            'num_attempts': len(all_results),
            'all_results': all_results
        }


def extract_mrz_advanced(image):
    """Main entry point for advanced MRZ extraction."""
    extractor = AdvancedMRZExtractor()
    return extractor.ensemble_extract(image)


# Check availability
def get_available_engines():
    """Return list of available OCR engines."""
    engines = []
    if TESSERACT_AVAILABLE:
        engines.append('Tesseract')
    if EASYOCR_AVAILABLE:
        engines.append('EasyOCR')
    return engines


if __name__ == "__main__":
    engines = get_available_engines()
    if engines:
        print(f"✅ Available OCR engines: {', '.join(engines)}")
    else:
        print("❌ No OCR engines available")
        print("Install: pip install pytesseract easyocr")
