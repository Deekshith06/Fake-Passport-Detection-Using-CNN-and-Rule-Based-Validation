# 🚀 Quick Start Guide

## 📦 Installation

### 1. System Requirements
- Python 3.8 or higher
- pip package manager
- Tesseract OCR (for MRZ extraction)

### 2. Install Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### 3. Install Python Dependencies

```bash
cd passport-verification-system

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# Install all dependencies
pip install -r requirements.txt
```

**Note:** First run may take time as EasyOCR downloads models (~100MB)

---

## 🎯 Running the Application

```bash
streamlit run app/streamlit_app.py
```

Open browser: http://localhost:8501

---

## 📖 How to Use

### **Method 1: Quick Verification (Manual MRZ)**

1.  **Go to:** "🔍 Verify Passport" page
2.  **Upload** passport image
3.  **Enter MRZ manually** (2 lines from passport bottom)
4.  **Click outside** text area to validate
5.  **View results:** MRZ validation → CNN analysis → Final verdict

### **Method 2: Automatic Extraction + Verification**

1.  **Go to:** "📄 MRZ Extractor" page
2.  **Upload** passport image
3.  **Click** "🚀 Extract MRZ"
4.  **Review** extracted text (edit if needed)
5.  **Download** or **Validate** immediately
6.  Or **copy** MRZ and use in Verify Passport page

---

## ✅ Expected Results

**For GENUINE Passport:**
- ✅ All MRZ checksums VALID
- ✅ CNN confidence > 0.5
- ✅ Final Verdict: GENUINE

**For FAKE Passport:**
- ❌ MRZ checksum FAILED → Immediate FAKE verdict
- (CNN not checked if MRZ fails)

---

## 🎓 For Viva/Demo

**Best Demo Flow:**
1. Show **MRZ Extractor** page - demonstrate OCR capability
2. Show **Verify Passport** - demonstrate validation logic
3. Explain **MRZ > CNN** hierarchy (checksum failure = definitive)
4. Show **About** page - explain technical approach

**Key Points:**
- MRZ validation is mathematical (definitive)
- CNN is probabilistic (supportive)
- Correct hierarchy prevents false positives

---

## 📚 Sample MRZ for Testing

**Polish Passport (GENUINE):**
```
P<POLMUSIELAK<<<BORYS<ANDRZEJ<<<<<<<<<<<<
EM9638245<POL8404238M3301256754<<<<<<<<<<<<2
```

**US Passport (Sample):**
```
P<USAERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<
L898902C36UTO7408122F1204159ZE184226B<<<<<10
```

---

## ⚠️ Troubleshooting

**"Tesseract not found"**
- Install Tesseract OCR (see step 2 above)
- Restart terminal after installation

**"EasyOCR not working"**
- First run downloads models (takes time)
- Check internet connection
- Alternatively, uninstall: `pip uninstall easyocr`

**"Streamlit won't start"**
- Check virtual environment is activated
- Reinstall: `pip install -r requirements.txt --force-reinstall`

---

## 📄 Next Steps

- Read `INTERVIEW_QA.md` for viva preparation
- Check `MRZ_FIELDS_GUIDE.md` for MRZ technical details
- Review `TECHNICAL_REPORT.md` for complete documentation

---

## 📚 Key Files to Know

| File | Purpose |
|------|---------|
| `README.md` | Complete project documentation |
| `INTERVIEW_QA.md` | 20+ interview questions with answers |
| `TECHNICAL_REPORT.md` | Academic report template |
| `src/models/cnn_model.py` | CNN architecture code |
| `src/features/rule_based_checks.py` | Forensic validation |
| `notebooks/02_model_training.ipynb` | Training walkthrough |
| `app/streamlit_app.py` | Web interface |

---

## ⚡ Quick Commands

```bash
# Install everything
pip install -r requirements.txt

# Train model
jupyter notebook notebooks/02_model_training.ipynb

# Run web app
streamlit run app/streamlit_app.py

# Test single prediction (in Python)
python -c "from src.models.cnn_model import load_model, predict_single_image; \
           model = load_model('models/passport_cnn.h5'); \
           label, conf, score = predict_single_image(model, 'path/to/passport.jpg'); \
           print(f'{label}: {conf:.2%}')"
```

---

## ✅ Before Presentation Checklist

- [ ] Dataset collected (200+ images)
- [ ] Model trained successfully
- [ ] Web app runs without errors
- [ ] Take screenshots:
  - [ ] Training curves
  - [ ] Confusion matrix
  - [ ] Web app demo
- [ ] Fill your results in TECHNICAL_REPORT.md
- [ ] Add your name to all documents
- [ ] Practice demo (upload image → show prediction)
- [ ] Read INTERVIEW_QA.md (practice answers)
- [ ] Can explain CNN architecture from memory

---

## 🆘 Common Issues

**"No such file or directory: models/passport_cnn.h5"**
→ You haven't trained the model yet. Run notebook first!

**"TensorFlow not found"**
→ Run: `pip install tensorflow`

**"Tesseract not found"**
→ Install Tesseract OCR:
- macOS: `brew install tesseract`
- Ubuntu: `sudo apt-get install tesseract-ocr`
- Windows: Download from GitHub

**"Out of memory during training"**
→ In notebook, change `BATCH_SIZE = 16` to `BATCH_SIZE = 8`

**"Model predicts all REAL or all FAKE"**
→ Class imbalance issue. Check class_weights are applied in training.

---

## 📊 What Success Looks Like

**Good Results:**
- Accuracy: 80-90%
- Training/validation curves close together
- Confusion matrix shows balanced performance
- Web app makes reasonable predictions

**If Results Are Poor (<75%):**
- Collect more data
- Increase training epochs
- Check if fake images are realistic enough
- Verify class balance

---

## 🎯 Your Goal

**Demonstrate you understand:**
1. What CNN is and why it's used
2. Transfer learning concept
3. How hybrid system works (CNN + rules)
4. Class imbalance handling
5. Overfitting prevention
6. Model evaluation metrics

**Show working demo:**
1. Upload real passport → System says REAL
2. Upload fake passport → System says FAKE
3. Explain why (forensic checks)

---

## 📞 Need Help?

**For code issues:**
- Read comments in the .py files
- Check README troubleshooting section

**For concepts:**
- Review INTERVIEW_QA.md
- Search YouTube: "CNN explained", "Transfer learning"

**For presentation:**
- Practice 5-min demo
- Memorize architecture diagram
- Be ready to explain any code section

---

**You've got this! 🎓**

The hardest part (building the system) is done.  
Now just:
1. Get dataset
2. Train model
3. Practice explaining

Good luck! 🚀
