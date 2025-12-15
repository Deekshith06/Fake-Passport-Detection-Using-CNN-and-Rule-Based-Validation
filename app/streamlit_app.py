"""
PASSPORT VERIFICATION SYSTEM - STREAMLIT APP
============================================
Clean, error-free implementation with beautiful UI

Student Project: MRZ-First Validation System
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.data.preprocessing import load_and_preprocess_image
    from src.models.cnn_model import load_model, predict_single_image
    from src.features.rule_based_checks import run_all_rules
except ImportError as e:
    st.error(f"Import error: {e}")


# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="Passport Verification System",
    page_icon="🛂",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===== SIMPLIFIED CSS (FIXED FOR DARK THEME) =====
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 2rem 0;
        color: #667eea;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #94A3B8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .result-box {
        padding: 2.5rem;
        border-radius: 1.5rem;
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 2rem 0;
    }
    
    .real-result {
        background-color: #D1FAE5;
        color: #065F46;
        border: 4px solid #10B981;
    }
    
    .fake-result {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 4px solid #EF4444;
    }
    
    .info-box {
        background-color: rgba(59, 130, 246, 0.2);
        border-left: 4px solid #3B82F6;
        padding: 1.2rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: rgba(245, 158, 11, 0.2);
        border-left: 4px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: rgba(16, 185, 129, 0.2);
        border-left: 4px solid #10B981;
        padding: 1.2rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
    }
    
    .stage-header {
        background-color: #3B82F6;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 0.75rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ===== HELPER FUNCTIONS =====

def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV format."""
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    return np.array(pil_image)


@st.cache_resource
def load_trained_model():
    """Load CNN model (cached)."""
    try:
        model_path = "models/passport_cnn.h5"
        if os.path.exists(model_path):
            return load_model(model_path)
    except Exception:
        pass
    return None


# ===== MAIN APP =====

def main():
    # Header
    st.markdown('<h1 class="main-header">🛂 Passport Verification System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">MRZ-First Validation | Educational Demo Project</p>', unsafe_allow_html=True)
    
    # Educational disclaimer
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ EDUCATIONAL PROJECT</strong><br>
        This system demonstrates MRZ checksum validation and CNN concepts for educational purposes only.
        <strong>Not for real security use.</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("� Navigation")
        
        page = st.radio(
            "Go to:",
            ["🔍 Verify Passport", "� MRZ Extractor", "ℹ️ About Project", "⚙️ How It Works"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Settings section (only for verify page)
        if page == "🔍 Verify Passport":
            st.markdown("### ⚙️ Settings")
            
            use_rules = st.checkbox(
                "Enable Forensic Checks",
                value=True,
                help="Check text spacing, background patterns, etc."
            )
            
            threshold = st.slider(
                "CNN Decision Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Higher = more strict (fewer false positives)"
            )
        else:
            use_rules = True
            threshold = 0.5
    
    # ===== PAGE 1: VERIFY PASSPORT =====
    if page == "🔍 Verify Passport":
        st.markdown('<div class="stage-header">📤 Upload Passport Image</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a passport image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo of a passport"
        )
        
        if uploaded_file is not None:
            # Load image
            pil_image = Image.open(uploaded_file)
            img_array = pil_to_cv2(pil_image)
            
            # Create columns for layout
            col1, col2 = st.columns([1, 1], gap="large")
            
            # === LEFT COLUMN ===
            with col1:
                st.markdown("#### 📷 Uploaded Image")
                st.image(pil_image, use_container_width=True)
                
                st.markdown("---")
                st.markdown("#### 📝 Enter MRZ Text")
                
                st.info("""
                **💡 Tip:** For automatic extraction, use the **📄 MRZ Extractor** page from the sidebar.
                
                This page focuses on validation after you have the MRZ text.
                """)
                
                mrz_input = st.text_area(
                    "MRZ Text (2 lines from bottom of passport):",
                    height=90,
                    placeholder="P<POLMUSIELAK<<<BORYS<ANDRZEJ<<<<<<<<<<<<\nEM9638245<POL8404238M3301256754<<<<<<<<<<<<2",
                    help="Enter the 2 lines of MRZ text from the passport bottom",
                    key="mrz_input_box"
                )
                
                with st.expander("📋 Sample MRZ Format Reference"):
                    st.code("""Polish Passport:
P<POLMUSIELAK<<<BORYS<ANDRZEJ<<<<<<<<<<<<
EM9638245<POL8404238M3301256754<<<<<<<<<<<<2

US Passport:
P<USAERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<
L898902C36UTO7408122F1204159ZE184226B<<<<<10
                    
Note: Each line must be exactly 44 characters""", language="text")
            
            # === RIGHT COLUMN ===
            with col2:
                if mrz_input and len(mrz_input.strip()) > 20:
                    # Ensure MRZ is 88 characters
                    mrz_lines = mrz_input.strip().split('\n')
                    if len(mrz_lines) >= 2:
                        line1 = mrz_lines[0]
                        line2 = mrz_lines[1]
                    else:
                        line1 = mrz_input[:44] if len(mrz_input) >= 44 else mrz_input
                        line2 = mrz_input[44:] if len(mrz_input) > 44 else ""
                    
                    # Pad to 44 characters
                    while len(line1) < 44:
                        line1 += "<"
                    while len(line2) < 44:
                        line2 += "<"
                    
                    full_mrz = line1 + line2
                    
                    # STAGE 1: MRZ EXTRACTION & VALIDATION
                    st.markdown('<div class="stage-header">📋 Stage 1: MRZ Analysis</div>', unsafe_allow_html=True)
                    
                    with st.spinner("🔍 Analyzing MRZ..."):
                        # Import complete parser
                        from src.features.complete_mrz_parser import parse_mrz_fields, verify_all_checksums
                        
                        # Extract all fields
                        fields = parse_mrz_fields(full_mrz)
                        
                        # Check if parsing succeeded
                        if 'error' not in fields:
                            # Run checksum verification FIRST
                            checksums = verify_all_checksums(fields)
                            
                            # Update status message based on validation (NOT extraction)
                            st.markdown("### ✅ Step 2: MRZ Validation")
                            if checksums['overall_valid']:
                                st.success("✅ All checksums verified - MRZ data is intact")
                            else:
                                st.error("❌ Checksum validation failed - Data has been tampered")
                            
                            # Show extracted information in expandable sections
                            with st.expander("📄 Extracted Document Information"):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Document Type", f"{fields['document_type_name']}")
                                    st.metric("Passport Number", f"{fields['passport_number']}")
                                with col_b:
                                    st.metric("Issuing Country", f"{fields['issuing_country']}")
                                    st.metric("Nationality", f"{fields['nationality']}")
                            
                            with st.expander("👤 Personal Information"):
                                st.write(f"**Name:** {fields['surname']}, {fields['given_names']}")
                                col_c, col_d, col_e = st.columns(3)
                                with col_c:
                                    st.metric("Date of Birth", fields['date_of_birth'])
                                with col_d:
                                    st.metric("Gender", fields['gender'])
                                with col_e:
                                    st.metric("Expiry Date", fields['expiry_date'])
                            
                            # Show detailed checksum verification
                            with st.expander("🔢 Checksum Verification Details", expanded=True):
                                passed_count = sum(1 for name, r in checksums.items() if name != 'overall_valid' and r['valid'])
                                total_count = len([k for k in checksums.keys() if k != 'overall_valid'])
                                st.write(f"**Status: {passed_count}/{total_count} checksums passed**")
                                
                                st.markdown("---")
                                for field_name, result in checksums.items():
                                    if field_name == 'overall_valid':
                                        continue
                                    
                                    col_x, col_y, col_z = st.columns([2, 1, 1])
                                    with col_x:
                                        if result['valid']:
                                            st.success(f"✅ {field_name.upper().replace('_', ' ')}")
                                        else:
                                            st.error(f"❌ {field_name.upper().replace('_', ' ')}")
                                    with col_y:
                                        st.text(f"Calc: {result['calculated']}")
                                    with col_z:
                                        st.text(f"Print: {result['printed']}")
                            
                            # === CRITICAL DECISION POINT ===
                            st.markdown("---")
                            st.markdown("### 🎯 Final Verdict")
                            
                            # MRZ CHECKSUM IS PRIMARY DECISION MAKER
                            if not checksums['overall_valid']:
                                # ❌ MRZ FAILED → IMMEDIATE FAKE VERDICT
                                st.markdown("""
                                <div class="result-box fake-result">
                                    ❌ FAKE PASSPORT
                                    <br><span style="font-size: 1.2rem;">MRZ Checksum Validation FAILED</span>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Show which checksums failed
                                failed_checks = [name.replace('_', ' ').upper() for name, result in checksums.items() 
                                               if name != 'overall_valid' and not result['valid']]
                                
                                st.error(f"**Failed Checksums:** {', '.join(failed_checks)}")
                                st.warning("""
                                **Decision Logic:** MRZ checksum failure provides mathematical proof of tampering.  
                                **Result:** FAKE - regardless of visual appearance.  
                                **No CNN check needed** - checksum failure is definitive.
                                """)
                            
                            else:
                                # ✅ MRZ PASSED → Check CNN (optional secondary verification)
                                st.success("✅ **MRZ Validation Passed** - All checksums mathematically correct")
                                
                                # Optional CNN check for additional confidence
                                model = load_trained_model()
                                
                                if model and use_rules:
                                    st.markdown("---")
                                    st.markdown("#### 🤖 Stage 2: CNN Visual Verification (Optional)")
                                    st.info("MRZ passed. Running CNN for additional visual analysis...")
                                    
                                    temp_path = "temp.jpg"
                                    pil_image.save(temp_path)
                                    
                                    with st.spinner("Analyzing visual features..."):
                                        label, confidence, cnn_score = predict_single_image(model, temp_path, threshold)
                                    
                                    st.metric("CNN Confidence", f"{cnn_score:.3f}", help="0=Fake, 1=Real")
                                    st.progress(float(cnn_score))
                                    
                                    if cnn_score < threshold:
                                        # CNN says FAKE (but MRZ is valid)
                                        st.markdown("""
                                        <div class="result-box fake-result">
                                            ❌ SUSPICIOUS PASSPORT
                                            <br><span style="font-size: 1.2rem;">CNN detected visual anomalies</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        st.warning(f"**Decision:** MRZ is valid, but CNN score ({cnn_score:.3f}) suggests visual inconsistencies. Recommend manual review.")
                                    else:
                                        # Both MRZ and CNN passed
                                        st.markdown("""
                                        <div class="result-box real-result">
                                            ✅ GENUINE PASSPORT
                                            <br><span style="font-size: 1.2rem;">All validations passed ✓</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        st.success(f"**Decision:** MRZ checksums valid + CNN confidence {cnn_score:.3f} = GENUINE")
                                    
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)
                                else:
                                    # MRZ passed, no CNN available
                                    st.markdown("""
                                    <div class="result-box real-result">
                                        ✅ GENUINE PASSPORT
                                        <br><span style="font-size: 1.2rem;">Based on MRZ Validation</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.info("**Decision:** All MRZ checksums passed. CNN not available but MRZ validation is sufficient.")
                            
                            # Run additional forensic checks for reference
                            rule_results = run_all_rules(img_array, mrz_text=full_mrz)
                            
                        else:
                            st.error(f"❌ MRZ Parsing Error: {fields['error']}")
                            st.info("Please check the MRZ format and try again.")
                
                else:
                    st.markdown('<div class="info-box">👈 <strong>Please enter MRZ text in the box on the left</strong><br>Copy the two lines of text from the bottom of the passport image.</div>', unsafe_allow_html=True)
    
    # ===== PAGE 2: ABOUT =====
    
    # ===== PAGE 2: MRZ EXTRACTOR =====
    elif page == "� MRZ Extractor":
        st.markdown('\u003cdiv class="main-header"\u003e📄 Advanced MRZ Extractor\u003c/div\u003e', unsafe_allow_html=True)
        st.markdown('\u003cdiv class="subtitle"\u003eHigh-Accuracy MRZ Extraction using Multiple OCR Engines\u003c/div\u003e', unsafe_allow_html=True)
        
        # Info about this page
        st.info("""
        **🎯 Purpose:** This tool uses advanced OCR techniques to extract MRZ text with maximum accuracy.
        
        **🔧 Methods Used:**
        - Multiple preprocessing techniques (CLAHE, denoising, sharpening)
        - Ensemble OCR (Tesseract + EasyOCR)
        - Voting system for best results
        
        **⚡ Best For:** Extracting MRZ from high-quality scans or photos
        """)
        
        # Check available OCR engines
        from src.features.advanced_mrz_extraction import get_available_engines
        available_engines = get_available_engines()
        
        if not available_engines:
            st.error("""
            ❌ **No OCR engines available!**
            
            Please install at least one:
            ```bash
            pip install pytesseract  # Tesseract
            pip install easyocr      # EasyOCR (deep learning)
            ```
            """)
            st.stop()
        
        st.success(f"✅ Available OCR engines: **{', '.join(available_engines)}**")
        
        # File upload
        st.markdown("### 📤 Upload Passport Image")
        uploaded_file = st.file_uploader(
            "Choose a passport image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of passport (MRZ should be visible at bottom)"
        )
        
        if uploaded_file is not None:
            # Load image
            pil_image = Image.open(uploaded_file)
            img_array = pil_to_cv2(pil_image)
            
            # Display image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 📷 Original Image")
                st.image(pil_image, use_container_width=True)
            
            with col2:
                st.markdown("#### 🔍 MRZ Region (Bottom 28%)")
                h = img_array.shape[0]
                mrz_region = img_array[int(h * 0.72):h, :]
                st.image(cv2.cvtColor(mrz_region, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Extract button
            st.markdown("---")
            if st.button("🚀 Extract MRZ", type="primary", use_container_width=True):
                with st.spinner("Extracting MRZ using all available methods..."):
                    from src.features.advanced_mrz_extraction import extract_mrz_advanced
                    
                    result = extract_mrz_advanced(img_array)
                    
                    if result['success']:
                        # Show results
                        st.markdown("### ✅ Extraction Successful!")
                        
                        # Main result
                        st.success(f"**Engine Used:** {result['engine']}")
                        st.metric("Confidence", f"{result['confidence']:.0%}")
                        st.info(f"**Total Attempts:** {result['num_attempts']} different methods tried")
                        
                        # Display extracted MRZ
                        st.markdown("#### 📝 Extracted MRZ Text:")
                        mrz_text = st.text_area(
                            "MRZ (Editable):",
                            value=result['mrz_text'],
                            height=80,
                            key="extracted_mrz"
                        )
                        
                        # Show all attempts
                        with st.expander("🔍 View All Extraction Attempts"):
                            for idx, attempt in enumerate(result['all_results'], 1):
                                st.markdown(f"**Attempt {idx}:** {attempt['engine']} (Confidence: {attempt['confidence']:.1%})")
                                st.code(f"{attempt['line1']}\n{attempt['line2']}", language="text")
                        
                        # Copy/Download buttons
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("📋 Copy to Clipboard"):
                                st.code(mrz_text, language="text")
                                st.success("✅ MRZ text displayed above - copy manually")
                        
                        with col_b:
                            st.download_button(
                                label="💾 Download MRZ",
                                data=mrz_text,
                                file_name="extracted_mrz.txt",
                                mime="text/plain"
                            )
                        
                        # Option to validate
                        st.markdown("---")
                        if st.button("✅ Validate This MRZ", use_container_width=True):
                            from src.features.complete_mrz_parser import parse_mrz_fields, verify_all_checksums
                            
                            # Pad MRZ to 88 characters
                            full_mrz = mrz_text.replace('\n', '').replace('\r', '')
                            while len(full_mrz) < 88:
                                full_mrz += '<'
                            
                            fields = parse_mrz_fields(full_mrz)
                            
                            if 'error' not in fields:
                                checksums = verify_all_checksums(fields)
                                
                                st.markdown("#### 📊 Validation Results:")
                                
                                if checksums['overall_valid']:
                                    st.success("✅ All checksums VALID - MRZ is intact!")
                                else:
                                    st.error("❌ Checksum validation FAILED!")
                                
                                # Show details
                                for name, result in checksums.items():

                                    if name == 'overall_valid':
                                        continue
                                    status = "✅" if result['valid'] else "❌"
                                    st.write(f"{status} {name.replace('_', ' ').title()}: Calc={result['calculated']}, Print={result['printed']}")
                    else:
                        st.error(f"❌ {result['message']}")
        
        # Installation help
        st.markdown("---")
        with st.expander("�📚 Installation Guide"):
            st.markdown("""
            ### Install OCR Engines
            
            **Tesseract:**
            ```bash
            # macOS
            brew install tesseract
            pip install pytesseract
            
            # Ubuntu/Debian
            sudo apt-get install tesseract-ocr
            pip install pytesseract
            ```
            
            **EasyOCR (Deep Learning):**
            ```bash
            pip install easyocr
            ```
            
            **Note:** EasyOCR will download models (~100MB) on first use.
            """)
    
    # ===== PAGE 3: ABOUT PROJECT =====
    elif page == "ℹ️ About Project":
        st.markdown('<div class="stage-header">📚 About This Project</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Project Overview
        
        This **student-level educational project** demonstrates passport verification using:
        - **Stage 1:** MRZ checksum validation (ICAO 9303)
        - **Stage 2:** CNN visual analysis (optional)
        
        ### 🏗️ Technical Stack
        - **Framework:** Streamlit (Web UI)
        - **Deep Learning:** TensorFlow/Keras
        - **Model:** EfficientNetB0 (transfer learning)
        - **Validation:** MRZ checksum algorithm
        
        ### 🎓 Learning Objectives
        - Image preprocessing techniques
        - MRZ checksum mathematics
        - CNN architecture understanding
        - Hybrid AI systems (ML + Rules)
        
        ### ⚠️ Limitations
        1. Visual-only detection (no database)
        2. No RFID chip validation
        3. Training data dependent
        4. Educational use only
        """)
    
    # ===== PAGE 3: HOW IT WORKS =====
    else:
        st.markdown('<div class="stage-header">⚙️ How It Works</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🔄 Verification Flow
        
        **Step 1: Upload Image** → Upload passport photo
        
        **Step 2: Enter MRZ** → Manually paste MRZ text
        
        **Step 3: Stage 1 - MRZ Validation** ✅
        ```
        Extract passport number from MRZ
        ↓
        Calculate checksum using ICAO 9303 algorithm
        ↓
        Compare with printed checksum
        ↓
        IF FAIL → Immediate FAKE verdict ❌
        IF PASS → Proceed to Stage 2 ✓
        ```
        
        **Step 4: Stage 2 - CNN Check** (if MRZ passed)
        ```
        Load pre-trained CNN model
        ↓
        Analyze visual features
        ↓
        Generate confidence score (0-1)
        ↓
        Classify as REAL or FAKE
        ```
        
        ### 🔢 MRZ Checksum Algorithm
        
        **Example:** Passport Number `EM9638245`
        
        ```
        Characters:  E  M  9  6  3  8  2  4  5
        Values:     14 22  9  6  3  8  2  4  5
        Weights:    ×7 ×3 ×1 ×7 ×3 ×1 ×7 ×3 ×1
        Products:   98 66  9 42  9  8 14 12  5
        
        Sum = 263
        Checksum = 263 % 10 = 3
        ```
        
        ### 🎯 Why This Approach?
        
        **MRZ First (Fast & Deterministic)**
        - Mathematical validation
        - No false positives
        - Quick decision
        
        **CNN Second (Visual Confirmation)**
        - Catches sophisticated fakes
        - Learns patterns humans miss
        - Extra layer of security
        """)


if __name__ == "__main__":
    main()
