"""
COMPREHENSIVE MRZ FIELD EXTRACTION
===================================
Extract ALL important fields from passport MRZ for verification.

Student-friendly implementation with clear explanations.
"""

def parse_mrz_fields(mrz_text):
    """
    Extract all important fields from MRZ text.
    
    MRZ Format (TD-3, 2 lines of 44 characters each):
    Line 1: Type + Country + Name
    Line 2: Passport# + Checksums + DOB + Gender + Expiry + Personal# + Final Checksum
    
    Args:
        mrz_text: Full MRZ string (88 characters)
        
    Returns:
        dict: All extracted fields with their values
    """
    
    if not mrz_text or len(mrz_text) < 88:
        return {"error": "Invalid MRZ length"}
    
    # Split into two lines
    line1 = mrz_text[:44]
    line2 = mrz_text[44:88]
    
    # ===== LINE 1 EXTRACTION =====
    
    # 1. Document Type (position 0)
    doc_type = line1[0]
    doc_type_name = {
        'P': 'Passport',
        'V': 'Visa',
        'I': 'ID Card'
    }.get(doc_type, 'Unknown')
    
    # 2. Issuing Country (positions 2-4)
    issuing_country = line1[2:5]
    
    # 3. Name (positions 5-43)
    name_field = line1[5:44].replace('<', ' ').strip()
    # Split surname and given names
    name_parts = name_field.split('  ')
    surname = name_parts[0] if name_parts else ''
    given_names = name_parts[1] if len(name_parts) > 1 else ''
    
    # ===== LINE 2 EXTRACTION =====
    
    # 4. Passport Number (positions 0-8)
    passport_number = line2[0:9].replace('<', '')
    
    # 5. Passport Number Checksum (position 9)
    passport_checksum = line2[9]
    
    # 6. Nationality (positions 10-12)
    nationality = line2[10:13]
    
    # 7. Date of Birth (positions 13-18)
    dob_raw = line2[13:19]
    dob_formatted = format_date(dob_raw)
    
    # 8. DOB Checksum (position 19)
    dob_checksum = line2[19]
    
    # 9. Gender (position 20)
    gender_code = line2[20]
    gender = {
        'M': 'Male',
        'F': 'Female',
        '<': 'Unspecified'
    }.get(gender_code, 'Unknown')
    
    # 10. Expiry Date (positions 21-26)
    expiry_raw = line2[21:27]
    expiry_formatted = format_date(expiry_raw)
    
    # 11. Expiry Checksum (position 27)
    expiry_checksum = line2[27]
    
    # 12. Personal Number (positions 28-41)
    personal_number = line2[28:42].replace('<', '').strip()
    personal_number = personal_number if personal_number else 'Not used'
    
    # 13. Final Checksum (position 43)
    final_checksum = line2[43]
    
    # Return all extracted fields
    return {
        # Line 1 fields
        'document_type': doc_type,
        'document_type_name': doc_type_name,
        'issuing_country': issuing_country,
        'surname': surname,
        'given_names': given_names,
        
        # Line 2 fields
        'passport_number': passport_number,
        'passport_checksum': passport_checksum,
        'nationality': nationality,
        'date_of_birth': dob_formatted,
        'dob_raw': dob_raw,
        'dob_checksum': dob_checksum,
        'gender': gender,
        'gender_code': gender_code,
        'expiry_date': expiry_formatted,
        'expiry_raw': expiry_raw,
        'expiry_checksum': expiry_checksum,
        'personal_number': personal_number,
        'final_checksum': final_checksum,
        
        # Raw lines for reference
        'line1': line1,
        'line2': line2
    }


def format_date(date_str):
    """
    Format YYMMDD to DD-MM-YYYY.
    
    Args:
        date_str: Date in YYMMDD format
        
    Returns:
        str: Formatted date DD-MM-YYYY or error message
    """
    if len(date_str) != 6:
        return f"{date_str} (Invalid length)"
    
    try:
        yy = date_str[0:2]
        mm = date_str[2:4]
        dd = date_str[4:6]
        
        # Validate numeric
        yy_int = int(yy)
        mm_int = int(mm)
        dd_int = int(dd)
        
        # Determine century (if YY > 30, assume 19xx, else 20xx)
        year = f"19{yy}" if yy_int > 30 else f"20{yy}"
        
        return f"{dd}-{mm}-{year}"
        
    except ValueError:
        # Non-numeric characters (OCR error)
        return f"{date_str} (OCR Error - contains non-numeric characters)"


def calculate_checksum(data):
    """
    Calculate MRZ checksum using ICAO 9303 algorithm.
    
    Weighting: 7-3-1 pattern (repeating)
    
    Args:
        data: String to calculate checksum for
        
    Returns:
        int: Checksum digit (0-9)
    """
    weights = [7, 3, 1]
    total = 0
    
    for i, char in enumerate(data):
        # Convert character to numeric value
        if char.isdigit():
            value = int(char)
        elif char.isalpha():
            value = ord(char) - ord('A') + 10
        elif char == '<':
            value = 0
        else:
            value = 0
        
        # Apply weight
        weight = weights[i % 3]
        total += value * weight
    
    return total % 10


def verify_all_checksums(fields):
    """
    Verify ALL checksums in the MRZ.
    
    Returns:
        dict: Verification results for each checksum
    """
    results = {}
    
    # 1. Passport Number Checksum
    passport_calc = calculate_checksum(fields['line2'][0:9])
    passport_printed = fields['passport_checksum']
    results['passport_number'] = {
        'calculated': passport_calc,
        'printed': passport_printed,
        'valid': str(passport_calc) == passport_printed
    }
    
    # 2. Date of Birth Checksum
    dob_calc = calculate_checksum(fields['dob_raw'])
    dob_printed = fields['dob_checksum']
    results['date_of_birth'] = {
        'calculated': dob_calc,
        'printed': dob_printed,
        'valid': str(dob_calc) == dob_printed
    }
    
    # 3. Expiry Date Checksum
    expiry_calc = calculate_checksum(fields['expiry_raw'])
    expiry_printed = fields['expiry_checksum']
    results['expiry_date'] = {
        'calculated': expiry_calc,
        'printed': expiry_printed,
        'valid': str(expiry_calc) == expiry_printed
    }
    
    # 4. Final Checksum (most important!)
    # Combines: passport# + checksum + DOB + checksum + expiry + checksum
    final_data = fields['line2'][0:10] + fields['line2'][13:20] + fields['line2'][21:43]
    final_calc = calculate_checksum(final_data)
    final_printed = fields['final_checksum']
    results['final'] = {
        'calculated': final_calc,
        'printed': final_printed,
        'valid': str(final_calc) == final_printed
    }
    
    # Overall result
    all_valid = all(r['valid'] for r in results.values())
    results['overall_valid'] = all_valid
    
    return results


def display_extracted_fields(fields):
    """
    Format extracted fields for display.
    
    Returns:
        str: Formatted text showing all fields
    """
    output = []
    output.append("=" * 70)
    output.append("MRZ FIELD EXTRACTION RESULTS")
    output.append("=" * 70)
    
    output.append("\n📋 DOCUMENT INFORMATION")
    output.append(f"   Type: {fields['document_type_name']} ({fields['document_type']})")
    output.append(f"   Issuing Country: {fields['issuing_country']}")
    
    output.append("\n👤 PERSONAL INFORMATION")
    output.append(f"   Surname: {fields['surname']}")
    output.append(f"   Given Names: {fields['given_names']}")
    output.append(f"   Nationality: {fields['nationality']}")
    output.append(f"   Gender: {fields['gender']} ({fields['gender_code']})")
    
    output.append("\n🔢 PASSPORT DETAILS")
    output.append(f"   Passport Number: {fields['passport_number']}")
    output.append(f"   Date of Birth: {fields['date_of_birth']}")
    output.append(f"   Expiry Date: {fields['expiry_date']}")
    output.append(f"   Personal Number: {fields['personal_number']}")
    
    output.append("\n✅ CHECKSUMS")
    output.append(f"   Passport #: {fields['passport_checksum']}")
    output.append(f"   DOB: {fields['dob_checksum']}")
    output.append(f"   Expiry: {fields['expiry_checksum']}")
    output.append(f"   Final: {fields['final_checksum']}")
    
    output.append("\n" + "=" * 70)
    
    return "\n".join(output)


# ===== TEST EXAMPLE =====

if __name__ == "__main__":
    # Polish passport MRZ
    test_mrz = "P<POLMUSIELAK<<<BORYS<ANDRZEJ<<<<<<<<<<<<EM9638245<POL8404238M3301256754<<<<<<<<02<<<"
    
    # Pad to 88 characters
    while len(test_mrz) < 88:
        test_mrz += "<"
    
    print("Testing MRZ Extraction\n")
    
    # Extract all fields
    fields = parse_mrz_fields(test_mrz)
    
    # Display results
    print(display_extracted_fields(fields))
    
    # Verify checksums
    print("\n📊 CHECKSUM VERIFICATION")
    print("=" * 70)
    checksums = verify_all_checksums(fields)
    
    for field_name, result in checksums.items():
        if field_name == 'overall_valid':
            continue
        
        status = "✅ VALID" if result['valid'] else "❌ INVALID"
        print(f"{field_name.upper()}: {status}")
        print(f"  Calculated: {result['calculated']}, Printed: {result['printed']}")
    
    print("\n" + "=" * 70)
    if checksums['overall_valid']:
        print("✅ RESULT: GENUINE PASSPORT - All checksums valid!")
    else:
        print("❌ RESULT: FAKE PASSPORT - Checksum validation failed!")
    print("=" * 70)
