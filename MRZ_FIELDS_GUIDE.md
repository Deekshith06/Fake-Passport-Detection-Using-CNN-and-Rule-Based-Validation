# 📚 MRZ Field Extraction Guide

## For Student Project Report & Viva

This guide explains ALL important MRZ fields for your passport verification project.

---

## 🎯 What is MRZ?

**MRZ = Machine Readable Zone**

- Two lines of text at the bottom of passport
- Contains standardized, verifiable data
- Uses ICAO 9303 international standard
- Each field has a specific position

---

## 📋 MRZ Format (TD-3 Passports)

```
Line 1: 44 characters
P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<

Line 2: 44 characters
L898902C36UTO7408122F1204159ZE184226B<<<<<10
```

**Total: 88 characters**

---

## 🔍 ALL IMPORTANT FIELDS

### 1️⃣ Document Type
- **Position:** Line 1, Character 1
- **Example:** `P`
- **Values:** P (Passport), V (Visa), I (ID Card)
- **Why Important:** Confirms it's a passport document

### 2️⃣ Issuing Country Code
- **Position:** Line 1, Characters 3-5
- **Example:** `UTO`, `IND`, `POL`, `USA`
- **Format:** 3-letter ICAO code
- **Why Important:** Detects fake country codes

### 3️⃣ Name
- **Position:** Line 1, Characters 6-44
- **Format:** SURNAME << GIVEN NAMES
- **Example:** `MUSIELAK<<<BORYS<ANDRZEJ`
- **Why Important:** Identity verification

### 4️⃣ Passport Number ⭐
- **Position:** Line 2, Characters 1-9
- **Example:** `L898902C3`
- **Why Important:** Unique identity, used for checksum

### 5️⃣ Passport Number Checksum ⭐⭐ (VERY IMPORTANT)
- **Position:** Line 2, Character 10
- **Example:** `6`
- **Calculation:** ICAO 9303 algorithm (7-3-1 weighting)
- **Why Important:** Detects if passport number was tampered

### 6️⃣ Nationality
- **Position:** Line 2, Characters 11-13
- **Example:** `IND`, `UTO`
- **Why Important:** Must match issuing country

### 7️⃣ Date of Birth
- **Position:** Line 2, Characters 14-19
- **Format:** YYMMDD
- **Example:** `800201` → 01-02-1980
- **Why Important:** Age verification, checksum validation

### 8️⃣ DOB Checksum
- **Position:** Line 2, Character 20
- **Why Important:** Detects DOB alteration

### 9️⃣ Gender
- **Position:** Line 2, Character 21
- **Values:** M (Male), F (Female), < (Unspecified)
- **Why Important:** Consistency check

### 🔟 Expiry Date
- **Position:** Line 2, Characters 22-27
- **Format:** YYMMDD
- **Example:** `101223` → 23-12-2010
- **Why Important:** Check if passport is expired

### 1️⃣1️⃣ Expiry Checksum
- **Position:** Line 2, Character 28
- **Why Important:** Detects expiry date tampering

### 1️⃣2️⃣ Personal Number
- **Position:** Line 2, Characters 29-42
- **Example:** Usually `<<<<<<<<<<<<<<` (not used)
- **Why Important:** Can contain national ID

### 1️⃣3️⃣ Final Checksum ⭐⭐⭐ (MOST IMPORTANT!)
- **Position:** Line 2, Character 44 (last character)
- **Includes:** Passport#, DOB, Expiry Date + their checksums
- **Why Important:** Master verification - if this fails, passport is FAKE

---

## 🔢 Checksum Algorithm (ICAO 9303)

**Example:** Calculate checksum for `L898902C3`

```
Characters:  L  8  9  8  9  0  2  C  3
Values:     21  8  9  8  9  0  2 12  3
Weights:    ×7 ×3 ×1 ×7 ×3 ×1 ×7 ×3 ×1
Products:  147 24  9 56 27  0 14 36  3

Sum = 147+24+9+56+27+0+14+36+3 = 316
Checksum = 316 % 10 = 6
```

**Weight Pattern:** 7-3-1, 7-3-1, 7-3-1 (repeating)

**Character Values:**
- Numbers (0-9) → 0-9
- Letters (A-Z) → 10-35 (A=10, B=11, ... Z=35)
- Filler (<) → 0

---

## 🎯 How MRZ Helps Detect Fakes

| Check | Purpose | Importance |
|-------|---------|------------|
| ✅ Passport # checksum | Detects number tampering | HIGH |
| ✅ DOB checksum | Detects age alteration | MEDIUM |
| ✅ Expiry checksum | Detects validity tampering | MEDIUM |
| ✅ Final checksum | Master verification | **CRITICAL** |
| ✅ Country vs nationality | Cross-validation | MEDIUM |
| ✅ Expiry date | Expired passport check | HIGH |
| ✅ MRZ format | Font/layout quality | LOW |

---

## 📝 Viva Question Answers

### Q: "What fields do you extract from MRZ?"

**Answer:**
> "We extract passport number, nationality, date of birth, gender, expiry date, and all checksum digits. These fields are used for mathematical verification and cross-validation."

### Q: "Why is checksum important?"

**Answer:**
> "Checksums use the ICAO 9303 algorithm to mathematically verify that data hasn't been altered. If someone changes even one digit in the passport number, the checksum won't match, and we can detect it's fake."

### Q: "What is the final checksum?"

**Answer:**
> "The final checksum is the most important verification. It's calculated using the passport number, date of birth, expiry date, and their individual checksums. It's the master check that validates all critical data at once."

---

## 🧮 MRZ Parsing Code

See: `src/features/complete_mrz_parser.py`

**Key Functions:**
- `parse_mrz_fields(mrz_text)` → Extract all fields
- `verify_all_checksums(fields)` → Validate checksums
- `display_extracted_fields(fields)` → Show results

---

## 📊 For Your Report - Include This Diagram

```
┌─────────────────────────────────────────────────┐
│ Line 1: P < U T O E R I K S S O N < < A N N A   │
│         │   └─┘ └────────┘     └────────┘       │
│         │    │      │              │            │
│      Type  Country  Surname    Given Names     │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Line 2: L 8 9 8 9 0 2 C 3 6 U T O 8 0 0 2 0 1   │
│         └────────┘     │ └─┘ └───┘ │ └───┘     │
│              │      Check  Nat. DOB│ Check     │
│         Passport #        Date─────┘           │
└─────────────────────────────────────────────────┘
         2 F 1 2 0 4 1 5 9 Z E 1 8 4 2 2 6 B < 1 0
         │ └───┘ │ └──────────────┘  │         │
      Gender Exp. Check Personal#  Check  Final Check
```

---

## ✅ Summary for Examiner

**Tell them:**

> "Our system extracts 13 key fields from the MRZ and validates 4 checksums. The final checksum is most critical - if it fails, the passport is definitively fake. This gives us mathematical certainty, unlike visual-only detection which can be fooled by high-quality forgeries."

---

**Perfect for:**
- ✅ Project Report
- ✅ Viva Defense
- ✅ Technical Presentation
- ✅ Interview Questions
