"""
RULE-BASED VALIDATION MODULE
=============================
This module implements simple forensic checks for passport validation.

Student: [Your Name]
Project: Fake Passport Detection Using CNN
Purpose: Educational project demonstrating hybrid AI (CNN + Rules)

WHY RULE-BASED VALIDATION?
- CNNs detect visual anomalies, but some forgeries have logical errors
- Example: Altered MRZ (Machine Readable Zone) has invalid checksums
- Rules add transparency: we can explain WHY a passport failed
- Combines ML (black box) with logic (white box) = Hybrid AI

RULES IMPLEMENTED:
1. MRZ Checksum Validation (mathematical verification)
2. Text Spacing Analysis (font consistency)
3. Background Pattern Check (FFT anomaly detection)
"""

import cv2
import numpy as np
from scipy import fft
import pytesseract


# ===== RULE 1: MRZ CHECKSUM VALIDATION =====

def validate_mrz_checksum(mrz_text):
    """
    Validate MRZ (Machine Readable Zone) checksum.
    
    WHAT IS MRZ?
    Two lines of text at bottom of passport:
    
    P<UTOeriksson<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<
    L898902C36UTO7408122F1204159ZE184226B<<<<<10
    
    Line 1: Passport type, country, name
    Line 2: Passport number, nationality, birth date, expiry, personal number
    
    WHAT IS A CHECKSUM?
    A calculated digit that verifies data integrity.
    Like a "signature" that proves data wasn't tampered with.
    
    HOW IT WORKS (ICAO 9303 Standard):
    1. Each character has a value:
       - Digits 0-9 → value is the digit itself
       - Letters A-Z → A=10, B=11, ..., Z=35
       - Filler '<' → 0
    
    2. Multiply each value by weight [7, 3, 1, 7, 3, 1, ...]
    
    3. Sum all products and take modulo 10
    
    4. Result should match the checksum digit
    
    EXAMPLE:
    Passport number: L898902C3
    Checksum: 6
    
    Calculation:
    L  8  9  8  9  0  2  C  3
    21 8  9  8  9  0  2  12 3
    ×7 ×3 ×1 ×7 ×3 ×1 ×7 ×3 ×1
    = 147+24+9+56+27+0+14+36+3 = 316
    316 % 10 = 6 ✓ (matches checksum!)
    
    Args:
        mrz_text (str): MRZ text (2 lines, 44 chars each)
        
    Returns:
        tuple: (bool: valid, str: calculation details)
    """
    
    def char_to_value(char):
        """Convert MRZ character to numerical value."""
        if char.isdigit():
            return int(char)
        elif char.isalpha():
            return ord(char.upper()) - ord('A') + 10
        else:  # '<' or other
            return 0
    
    def calculate_checksum(data):
        """Calculate checksum for given data string and return details."""
        weights = [7, 3, 1]
        total = 0
        details_lines = []
        
        details_lines.append(f"Passport Number: {data}")
        details_lines.append("-" * 60)
        details_lines.append("\nStep-by-step calculation:")
        
        for i, char in enumerate(data):
            value = char_to_value(char)
            weight = weights[i % 3]  # Cycle through weights
            product = value * weight
            total += product
            
            # Format calculation step
            if char.isalpha():
                details_lines.append(f"  Position {i}: '{char}' → {char}={value}, weight={weight}, product={product}")
            else:
                details_lines.append(f"  Position {i}: '{char}' → value={value}, weight={weight}, product={product}")
        
        checksum = total % 10
        details_lines.append(f"\nTotal sum = {total}")
        details_lines.append(f"Checksum = {total} % 10 = {checksum}")
        
        return checksum, "\n".join(details_lines)
    
    # Parse MRZ (simplified version)
    try:
        # Clean MRZ text
        mrz_text = mrz_text.replace(' ', '').replace('\n', '')
        
        if len(mrz_text) < 88:  # MRZ should be 2 lines × 44 chars = 88
            return False, "ERROR: MRZ text too short"
        
        # Line 2 contains the main checksums
        line2 = mrz_text[44:88]
        
        # Extract passport number (positions 0-8) and its checksum (position 9)
        passport_number = line2[0:9]
        passport_checksum = int(line2[9]) if line2[9].isdigit() else 0
        
        # Validate passport number checksum
        calculated, details = calculate_checksum(passport_number)
        
        details += f"\n\nPrinted checksum in MRZ: {passport_checksum}"
        details += f"\nCalculated checksum: {calculated}"
        
        if calculated != passport_checksum:
            details += f"\n\n❌ MISMATCH! Passport is FAKE"
            print(f"❌ MRZ checksum mismatch: calculated {calculated}, found {passport_checksum}")
            return False, details
        
        details += f"\n\n✅ MATCH! Checksum is valid"
        print(f"✓ MRZ checksum valid")
        return True, details
        
    except Exception as e:
        error_msg = f"❌ MRZ validation error: {e}"
        print(error_msg)
        return False, error_msg


# ===== RULE 2: TEXT SPACING ANALYSIS =====

def check_text_spacing(mrz_region_image):
    """
    Check if MRZ text has consistent character spacing.
    
    WHY THIS MATTERS:
    - Real passports use OCR-B font with exact spacing
    - Fake passports often use wrong fonts (Arial, Times, etc.)
    - Character spacing is a telltale sign
    
    HOW IT WORKS:
    1. Extract MRZ region from passport image
    2. Detect character bounding boxes
    3. Calculate spacing between characters
    4. Check if spacing is consistent
    
    THRESHOLDS:
    - Real passport: spacing variance < 3.0 pixels
    - Fake passport: spacing variance > 5.0 pixels
    
    Args:
        mrz_region_image: Cropped MRZ region (grayscale or color)
        
    Returns:
        bool: True if spacing is consistent, False otherwise
    """
    
    try:
        # Convert to grayscale if needed
        if len(mrz_region_image.shape) == 3:
            gray = cv2.cvtColor(mrz_region_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = mrz_region_image
        
        # Apply threshold to isolate text
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours (character boundaries)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get bounding boxes for each character
        bboxes = [cv2.boundingRect(c) for c in contours]
        
        # Sort by x-coordinate (left to right)
        bboxes = sorted(bboxes, key=lambda b: b[0])
        
        # Calculate spacing between characters
        spacings = []
        for i in range(len(bboxes) - 1):
            # Distance = start of next character - end of current character
            spacing = bboxes[i+1][0] - (bboxes[i][0] + bboxes[i][2])
            if spacing > 0:  # Only positive spacings
                spacings.append(spacing)
        
        if len(spacings) < 5:  # Need enough samples
            return False
        
        # Calculate variance in spacing
        spacing_variance = np.var(spacings)
        
        print(f"Text spacing variance: {spacing_variance:.2f} pixels")
        
        # Real passports have very consistent spacing
        if spacing_variance < 3.0:
            print("✓ Text spacing is consistent (likely REAL)")
            return True
        else:
            print("❌ Text spacing is irregular (possible FAKE)")
            return False
            
    except Exception as e:
        print(f"❌ Spacing check error: {e}")
        return False


# ===== RULE 3: BACKGROUND PATTERN CHECK (FFT) =====

def check_background_pattern(background_region):
    """
    Check guilloche pattern using FFT (Fast Fourier Transform).
    
    WHAT IS GUILLOCHE?
    Wavy line patterns in passport background (security feature).
    Real passports have mathematically precise patterns.
    
    WHAT IS FFT?
    Fast Fourier Transform - converts image from spatial domain to frequency domain.
    Think of it like analyzing sound:
    - Spatial domain: pixel positions
    - Frequency domain: wave patterns
    
    WHY THIS WORKS:
    - Real guilloche has specific frequency signatures
    - Scanned/reprinted passports lose frequency precision
    - FFT reveals if pattern is genuine or reproduced
    
    HOW IT WORKS:
    1. Convert background to grayscale
    2. Apply 2D FFT (find frequency components)
    3. Calculate power spectrum
    4. Check for strong frequency peaks
    
    INTERPRETATION:
    - Real: Sharp peaks at specific frequencies
    - Fake: Scattered frequencies, noise-like
    
    Args:
        background_region: Cropped background area (color or grayscale)
        
    Returns:
        bool: True if pattern looks genuine, False otherwise
    """
    
    try:
        # Convert to grayscale if needed
        if len(background_region.shape) == 3:
            gray = cv2.cvtColor(background_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = background_region
        
        # Apply 2D FFT
        fft_result = np.fft.fft2(gray)
        
        # Shift zero frequency to center
        fft_shifted = np.fft.fftshift(fft_result)
        
        # Calculate magnitude spectrum (power)
        magnitude = np.abs(fft_shifted)
        
        # Calculate power spectrum (squared magnitude)
        power_spectrum = magnitude ** 2
        
        # Find peak power (excluding DC component in center)
        h, w = power_spectrum.shape
        center_mask = np.ones_like(power_spectrum)
        center_mask[h//2-10:h//2+10, w//2-10:w//2+10] = 0  # Mask center
        
        masked_spectrum = power_spectrum * center_mask
        peak_power = np.max(masked_spectrum)
        mean_power = np.mean(masked_spectrum)
        
        # Calculate peak-to-mean ratio
        # Real passports have strong peaks (high ratio)
        # Fakes have scattered power (low ratio)
        ratio = peak_power / (mean_power + 1e-10)  # Avoid division by zero
        
        print(f"FFT peak-to-mean ratio: {ratio:.2f}")
        
        # Threshold determined empirically
        # Real passports: ratio > 50
        # Fake passports: ratio < 20
        if ratio > 50:
            print("✓ Background pattern has strong frequency signature (likely REAL)")
            return True
        else:
            print("❌ Background pattern lacks clear signature (possible FAKE)")
            return False
            
    except Exception as e:
        print(f"❌ FFT check error: {e}")
        return False


# ===== COMBINED RULE-BASED VALIDATION =====

def run_all_rules(passport_image, mrz_text=None):
    """
    Run all rule-based checks on a passport image.
    
    Args:
        passport_image: Full passport image (RGB)
        mrz_text: Extracted MRZ text (optional, will use OCR if None)
        
    Returns:
        dict: Results from all checks including MRZ details
    """
    
    print("\n" + "=" * 60)
    print("RUNNING RULE-BASED VALIDATION")
    print("=" * 60)
    
    results = {
        'mrz_valid': False,
        'spacing_consistent': False,
        'pattern_genuine': False,
        'overall_pass': False,
        'mrz_details': None
    }
    
    # Extract MRZ region (bottom 15% of image)
    h, w = passport_image.shape[:2]
    mrz_region = passport_image[int(h*0.85):, :]
    
    # Extract background region (middle area)
    bg_region = passport_image[int(h*0.3):int(h*0.7), int(w*0.1):int(w*0.9)]
    
    # Check 1: MRZ Checksum
    print("\n1. MRZ Checksum Validation:")
    if mrz_text:
        is_valid, details = validate_mrz_checksum(mrz_text)
        results['mrz_valid'] = is_valid
        results['mrz_details'] = details
    else:
        print("   ⚠ No MRZ text provided, skipping checksum validation")
        results['mrz_details'] = "⚠ No MRZ text provided"
    
    # Check 2: Text Spacing
    print("\n2. Text Spacing Analysis:")
    results['spacing_consistent'] = check_text_spacing(mrz_region)
    
    # Check 3: Background Pattern
    print("\n3. Background Pattern (FFT) Check:")
    results['pattern_genuine'] = check_background_pattern(bg_region)
    
    # Overall assessment
    # Passport passes if at least 2 out of 3 checks pass
    passed_checks = sum([
        results['mrz_valid'],
        results['spacing_consistent'],
        results['pattern_genuine']
    ])
    
    results['overall_pass'] = passed_checks >= 2
    
    print("\n" + "=" * 60)
    print(f"RULE-BASED VALIDATION RESULT: {'PASS ✓' if results['overall_pass'] else 'FAIL ✗'}")
    print(f"Checks passed: {passed_checks}/3")
    print("=" * 60)
    
    return results


# ===== MAIN EXECUTION (for testing) =====
if __name__ == "__main__":
    """
    Test code to verify rule-based checks work.
    """
    
    print("=" * 60)
    print("TESTING RULE-BASED VALIDATION MODULE")
    print("=" * 60)
    
    # Test MRZ checksum
    print("\nTest 1: MRZ Checksum Validation")
    
    # Valid MRZ example
    valid_mrz = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<L898902C36UTO7408122F1204159ZE184226B<<<<<10"
    result = validate_mrz_checksum(valid_mrz)
    print(f"Valid MRZ result: {result}")
    
    # Invalid MRZ example (altered)
    invalid_mrz = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<L898902C99UTO7408122F1204159ZE184226B<<<<<10"
    result = validate_mrz_checksum(invalid_mrz)
    print(f"Invalid MRZ result: {result}")
    
    print("\n" + "=" * 60)
    print("Rule-based validation module is ready to use!")
    print("=" * 60)
