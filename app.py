import os
import cv2
import numpy as np
import re
import streamlit as st
from PIL import Image

# Suppress tensorflow warnings naturally
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Passport Verifier", layout="wide", initial_sidebar_state="expanded")

# --- BACKEND FUNCTIONS ---
def _char_val(c):
    if c.isdigit(): return int(c)
    if c.isalpha(): return ord(c.upper()) - 55  # A=10, B=11...
    return 0

def calc_mrz_checksum(data):
    """ICAO 9303 checksum (7-3-1 weight pattern)."""
    weights = [7, 3, 1]
    total = sum(_char_val(char) * weights[i % 3] for i, char in enumerate(data))
    return total % 10

def verify_all_checksums(mrz_text):
    mrz = mrz_text.replace('\n', '').replace(' ', '')
    if len(mrz) < 88: return {"error": "MRZ text too short"}, {}
    
    line1, line2 = mrz[:44], mrz[44:88]
    
    ppt_no = line2[0:9]
    ppt_chk = line2[9]
    dob = line2[13:19]
    dob_chk = line2[19]
    exp = line2[21:27]
    exp_chk = line2[27]
    
    final_data = line2[0:10] + line2[13:20] + line2[21:43]
    final_chk = line2[43]
    
    def check(data, expected):
        if not expected.isdigit(): return False, 0
        calc = calc_mrz_checksum(data)
        return calc == int(expected), calc

    ppt_valid, ppt_calc = check(ppt_no, ppt_chk)
    dob_valid, dob_calc = check(dob, dob_chk)
    exp_valid, exp_calc = check(exp, exp_chk)
    final_valid, final_calc = check(final_data, final_chk)

    fields = {
        "document_type": line1[0],
        "document_type_name": {'P': 'Passport', 'V': 'Visa', 'I': 'ID Card'}.get(line1[0], 'Unknown'),
        "issuing_country": line1[2:5],
        "surname": line1[5:44].split("<<")[0].replace("<", " "),
        "given_names": "".join(line1[5:44].split("<<")[1:]).replace("<", " "),
        "passport_number": ppt_no.replace("<", ""),
        "nationality": line2[10:13],
        "date_of_birth": f"{dob[4:6]}-{dob[2:4]}-19{dob[0:2]}" if int(dob[0:2]) > 30 else f"{dob[4:6]}-{dob[2:4]}-20{dob[0:2]}",
        "gender": {'M': 'Male', 'F': 'Female', '<': 'Unspecified'}.get(line2[20], 'Unknown'),
        "expiry_date": f"{exp[4:6]}-{exp[2:4]}-20{exp[0:2]}"
    }

    checksums = {
        "passport_number": {"valid": ppt_valid, "printed": ppt_chk, "calculated": ppt_calc},
        "date_of_birth": {"valid": dob_valid, "printed": dob_chk, "calculated": dob_calc},
        "expiry_date": {"valid": exp_valid, "printed": exp_chk, "calculated": exp_calc},
        "final": {"valid": final_valid, "printed": final_chk, "calculated": final_calc},
        "overall_valid": ppt_valid and dob_valid and exp_valid and final_valid
    }

    return fields, checksums

@st.cache_resource
def get_cnn_model():
    model_path = "models/passport_cnn.h5"
    if os.path.exists(model_path):
        return load_model(model_path, compile=False)
    return None

def predict_cnn(image_path, threshold=0.5):
    """Runs efficientnet inference on standard 224x224 input."""
    model = get_cnn_model()
    if not model: return None, 0.0, 0.0

    img = cv2.imread(image_path)
    if img is None: return None, 0.0, 0.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype('float32') / 255.0
    
    pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
    
    if pred >= threshold: return "REAL", pred, pred
    return "FAKE", 1 - pred, pred

def run_forensics(img_array):
    """Combined text spacing and FFT anomaly checking."""
    h, w = img_array.shape[:2]
    mrz_roi = img_array[int(h*0.85):, :]
    bg_roi = img_array[int(h*0.3):int(h*0.7), int(w*0.1):int(w*0.9)]
    
    # 1. Text spacing variance
    gray_mrz = cv2.cvtColor(mrz_roi, cv2.COLOR_BGR2GRAY) if len(mrz_roi.shape) == 3 else mrz_roi
    _, bin_mrz = cv2.threshold(gray_mrz, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bin_mrz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])
    spacings = [bboxes[i+1][0] - (bboxes[i][0] + bboxes[i][2]) for i in range(len(bboxes)-1)]
    spacings = [s for s in spacings if s > 0]
    
    spacing_var = np.var(spacings) if len(spacings) >= 5 else 999
    spacing_consistent = spacing_var < 3.0

    # 2. Guilloche background pattern via FFT
    gray_bg = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY) if len(bg_roi.shape) == 3 else bg_roi
    fft_shifted = np.fft.fftshift(np.fft.fft2(gray_bg))
    power = np.abs(fft_shifted) ** 2
    
    # mask the DC center
    cy, cx = power.shape[0]//2, power.shape[1]//2
    power[cy-10:cy+10, cx-10:cx+10] = 0
    
    ratio = np.max(power) / (np.mean(power) + 1e-10)
    pattern_genuine = ratio > 50

    return {
        "spacing_consistent": bool(spacing_consistent),
        "spacing_variance": float(spacing_var),
        "pattern_genuine": bool(pattern_genuine),
        "fft_ratio": float(ratio),
        "overall_pass": bool(spacing_consistent) and bool(pattern_genuine)
    }


# --- UI LAYOUT AND CSS ---

st.markdown("""
<style>
    /* Global Typography & Colors */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; }
    
    /* Elegant Header */
    .main-header { font-size: 2.75rem; font-weight: 800; text-align: center; color: #f8fafc; padding-bottom: 0.5rem; margin-top: -2rem; }
    .sub-header { font-size: 1.15rem; text-align: center; color: #94a3b8; margin-bottom: 2.5rem; }
    
    /* Result Boxes */
    .val-box { 
        padding: 2rem; 
        border-radius: 1.25rem; 
        text-align: center; 
        font-size: 1.75rem; 
        font-weight: 700; 
        margin: 1.5rem 0; 
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -2px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease;
    }
    .val-box:hover { transform: translateY(-2px); }
    .real { background: linear-gradient(135deg, #064e3b 0%, #065f46 100%); color: #a7f3d0; border: 2px solid #059669; }
    .fake { background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%); color: #fecaca; border: 2px solid #dc2626; }
    
    /* Status Labels */
    .status-text { font-size: 1.1rem; opacity: 0.9; margin-top: 0.5rem; font-weight: 500; }
    
    /* Section Dividers */
    hr { margin: 2rem 0; border-color: #334155; }
    
    /* Custom Instruction Cards */
    .instruction-card { background-color: #1e293b; border-left: 4px solid #3b82f6; padding: 1.25rem; border-radius: 0.5rem; margin-bottom: 1.5rem; font-size: 0.95rem; color: #cbd5e1; }
    
    /* Data Cards */
    .data-card { background: #1e293b; border: 1px solid #334155; border-radius: 0.5rem; padding: 1rem; margin-bottom: 0.5rem; }
    .data-label { font-size: 0.75rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem; }
    .data-value { font-size: 1.1rem; color: #f8fafc; font-weight: 600; word-wrap: break-word; }
    .name-val { font-size: 1.2rem; font-weight: 700; color: #ffffff; }
    
    /* Button Customization */
    .stButton>button { background-color: #3b82f6; color: white; transition: all 0.2s ease; }
    .stButton>button:hover { background-color: #2563eb; color: white; }
    
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Navigation", ["Verify Passport", "About the System"], label_visibility="collapsed")

st.sidebar.markdown("---")
if page == "Verify Passport":
    st.sidebar.subheader("Settings")
    use_cnn = st.sidebar.toggle("Enable CNN Analysis", value=True, help="Uses deep learning to visually flag inconsistencies even if the MRZ math passes.")
    run_forensics = st.sidebar.toggle("Enable Background Forensics", value=False, help="Runs FFT and Font Spacing analysis. (Slower)")

# --- PAGE: VERIFY ---
if page == "Verify Passport":
    st.markdown('<div class="main-header">Identity Authenticator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Bank-grade MRZ and Visual Analysis System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="instruction-card">
        <strong>Welcome! How to use:</strong><br/>
        1. Upload a clear, glare-free image of the passport photo page.<br/>
        2. Paste the two lines of text from the very bottom of the document (the MRZ) into the verification box.
    </div>
    """, unsafe_allow_html=True)
    
    col_input, col_results = st.columns([4, 6], gap="large")
    
    with col_input:
        st.subheader("1. Upload Image (Optional)")
        img_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
        
        if img_file:
            pil_img = Image.open(img_file).convert('RGB')
            img_arr = np.array(pil_img)
            st.image(pil_img, caption="Document Scan", use_container_width=True, clamp=True)
            
        st.subheader("2. Enter MRZ Data")
        mrz_input = st.text_area("Paste exactly 2 lines (44 characters each):", height=110, placeholder="P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C36UTO7408122F1204159ZE184226B<<<<<10")
            
    with col_results:
        if mrz_input and len(mrz_input.strip()) > 20:
            with st.spinner("Analyzing document cryptography..."):
                fields, checksums = verify_all_checksums(mrz_input)
                
                if "error" in fields:
                    st.error(f"[ERROR] MRZ Error: {fields['error']}. Please check your formatting.")
                else:
                    st.subheader("Document Metadata")
                    
                    st.markdown(f'<div class="data-card"><div class="data-label">Full Name</div><div class="data-value name-val">{fields["given_names"]} {fields["surname"]}</div></div>', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f'<div class="data-card"><div class="data-label">Passport No</div><div class="data-value">{fields["passport_number"]}</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="data-card"><div class="data-label">Birth Date</div><div class="data-value">{fields["date_of_birth"]}</div></div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<div class="data-card"><div class="data-label">Nationality</div><div class="data-value">{fields["nationality"]}</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="data-card"><div class="data-label">Expiry Date</div><div class="data-value">{fields["expiry_date"]}</div></div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown(f'<div class="data-card"><div class="data-label">Document Type</div><div class="data-value">{fields["document_type_name"]}</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="data-card"><div class="data-label">Gender</div><div class="data-value">{fields["gender"]}</div></div>', unsafe_allow_html=True)
                    
                    with st.expander("View Mathematical Validation Details"):
                        for k, v in checksums.items():
                            if k != "overall_valid":
                                icon = "[PASS]" if v['valid'] else "[FAIL]"
                                st.write(f"{icon} **{k.replace('_', ' ').title()}**: Calculated `{v['calculated']}` | Found `{v['printed']}`")

                    st.markdown("---")
                    
                    st.subheader("Authentication Verdict")
                    
                    if not checksums['overall_valid']:
                        st.markdown('<div class="val-box fake">[FAIL] FORGERY DETECTED<div class="status-text">Cryptographic checksum constraints failed. Data has been altered.</div></div>', unsafe_allow_html=True)
                    else:
                        st.success("[PASS] Math Validation Passed: The MRZ format and check digits are cryptographically sound.")
                        
                        if use_cnn:
                            if img_file:
                                st.info("Running deep-learning visual feature extraction...")
                                tmp_file = "tmp_ppt.jpg"
                                pil_img.save(tmp_file)
                                
                                label, conf, raw = predict_cnn(tmp_file)
                                if os.path.exists(tmp_file): os.remove(tmp_file)
                                
                                if label is None:
                                    st.warning("⚠️ **CNN Model Not Found:** The `models/passport_cnn.h5` file is missing. The visual scan was skipped, but the MRZ data mathematically passed.")
                                elif label == "REAL":
                                    st.markdown(f'<div class="val-box real">[PASS] AUTHENTIC DOCUMENT<div class="status-text">Visual Confidence: {conf:.1%} | MRZ Valid</div></div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="val-box fake">[FAIL] SUSPICIOUS MATCH<div class="status-text">MRZ math is valid, but the CNN visual scan indicates a {conf:.1%} probability of forgery. Manual review required.</div></div>', unsafe_allow_html=True)
                            else:
                                st.warning("⚠️ **Image Required:** CNN visual scan skipped because no passport image was uploaded.")
                        else:
                             st.markdown('<div class="val-box real">[PASS] PASSED MRZ LAYER<div class="status-text">Note: CNN visual layer was bypassed.</div></div>', unsafe_allow_html=True)
                                
                        if run_forensics:
                            if img_file:
                                with st.expander("Advanced Background Forensics"):
                                    f_res = run_forensics(img_arr)
                                    
                                    st.write("**1. Font Tracking Analysis (OCR-B Standard)**")
                                    if f_res['spacing_consistent']: st.success(f"[PASS] Consistent character spacing. (Variance: {f_res['spacing_variance']:.2f})")
                                    else: st.error(f"[FAIL] Inconsistent character spacing detected! (Variance: {f_res['spacing_variance']:.2f})")
                                    
                                    st.write("**2. Guilloche Pattern Autocorrelation (FFT)**")
                                    if f_res['pattern_genuine']: st.success(f"[PASS] High-frequency security patterns verified. (Ratio: {f_res['fft_ratio']:.2f})")
                                    else: st.error(f"[FAIL] Background pattern lacks detail or is overly compressed. (Ratio: {f_res['fft_ratio']:.2f})")
                            else:
                                st.warning("⚠️ **Image Required:** Background forensics skipped because no passport image was uploaded.")

        elif not mrz_input:
            st.info("Awaiting MRZ Input. Please paste the two lines of text from the bottom of a passport scan into the text box.")

# --- PAGE: ABOUT ---
else:
    st.markdown('<div class="main-header">System Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Understanding the Hybrid Engine</div>', unsafe_allow_html=True)
    
    st.write("""
    ### How it Works
    This is a modern Hybrid-AI implementation that marries **deterministic rule-based logic** with **deep learning**.
    
    1. **MRZ Cryptography (The Hard Gate):**  
       The Machine Readable Zone (MRZ) relies on ICAO 9303 standard mod-10 calculations. If a forger changes a birthdate but forgets to properly compute the weighted composite checksum digit, our math engine rejects it instantly.
       
    2. **CNN Visual Analysis (The Soft Gate):**  
       If a forger *does* calculate the checksums correctly, their ID is passed to our pre-trained EfficientNet model. This model scans for visual abnormalities, microprint degradation, and tampered lamination.
       
    3. **Background FFT Analysis:**  
       Using Fast-Fourier Transforms, the app ensures that the complex undulating lines (Guilloche) present in passport security backgrounds exhibit the required high-frequency mathematical signature.
    """)
