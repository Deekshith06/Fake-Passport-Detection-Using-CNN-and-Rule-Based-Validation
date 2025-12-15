# Sample Passport for Testing

## Polish Passport - MUSIELAK, BORYS ANDRZEJ

**File:** `sample_polish_passport.jpg`

### MRZ Text
```
Line 1: P<POLMUSIELAK<<<BORYS<ANDRZEJ<<<<<<<<<<<<
Line 2: EM9638245<POL8404238M3301256754<<<<<<<<02
```

### How to Use in Streamlit App

1. Upload `sample_polish_passport.jpg`
2. Copy the MRZ text above
3. Paste it in the "Enter MRZ Text" box
4. See validation results!

### Expected Results

**Passport Number:** EM9638245
**Checksum Digit:** < (position 10 in Line 2)

The system will calculate the checksum and show whether it's REAL or FAKE.
