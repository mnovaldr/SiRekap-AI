from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import re
import joblib
from fastapi.middleware.cors import CORSMiddleware

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
except ImportError:
    print("[WARNING] Sastrawi belum terinstall. Stemming tidak akan berjalan.")
    StemmerFactory = None

app = FastAPI(
    title="MCC AI Microservice",
    description="Microservice untuk prediksi otomatis Program SDGs berbasis NLP (TF-IDF + SVM)",
    version="1.0.0"
)

# Izinkan CORS agar Laravel bisa menembak API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi Sastrawi Stemmer
stemmer = None
if StemmerFactory:
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

# Global variables untuk model
tfidf_vectorizer = None
svm_model = None
mlb_classes = None

@app.on_event("startup")
async def load_models():
    global tfidf_vectorizer, svm_model, mlb_classes
    try:
        tfidf_vectorizer = joblib.load('tfidf_vectorizer_final.pkl')
        svm_model = joblib.load('svm_tuned_model.pkl')
        mlb_classes = joblib.load('mlb_classes_final.pkl')
        print("[AI] Berhasil memuat model SVM, TF-IDF, dan MLB!")
    except Exception as e:
        print(f"[AI ERROR] Gagal memuat model .pkl. Error: {e}")

class PredictRequest(BaseModel):
    judul_kegiatan: str
    nama_organisasi: str = ""

def clean_text(text: str) -> str:
    """Membersihkan dan melakukan stemming pada teks input."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if stemmer:
        text = stemmer.stem(text)
    return text

@app.post("/api/predict-sdgs")
async def predict_sdgs(req: PredictRequest):
    """
    Menerima Judul Kegiatan dan mengembalikan prediksi kategori SDGs
    menggunakan model SVM yang sudah dilatih.
    """
    print(f"\n[AI] Menerima request prediksi untuk: '{req.judul_kegiatan}'")
    
    # 1. Pastikan model sudah di-load
    if svm_model is None or tfidf_vectorizer is None or mlb_classes is None:
        return {
            "status": "error",
            "message": "Model AI belum siap atau gagal dimuat.",
            "predicted_sdgs": []
        }

    # 2. Bersihkan teks (Preprocessing)
    cleaned_text = clean_text(req.judul_kegiatan)
    print(f"[AI] Teks setelah dibersihkan: '{cleaned_text}'")

    # 3. TF-IDF Vectorization
    X_input = tfidf_vectorizer.transform([cleaned_text])
    
    # 4. Prediksi menggunakan SVM
    pred_biner = svm_model.predict(X_input)
    
    # 5. Inverse Transform (Mengubah array 1 dan 0 kembali menjadi teks SDGS X)
    predicted_labels = mlb_classes.inverse_transform(pred_biner)
    
    predicted_sdgs = []
    if predicted_labels and len(predicted_labels[0]) > 0:
        predicted_sdgs = list(predicted_labels[0])
        
    print(f"[AI] Hasil prediksi SVM: {predicted_sdgs}")
    
    return {
        "status": "success",
        "input": req.judul_kegiatan,
        "predicted_sdgs": predicted_sdgs
    }

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Microservice AI aktif dan siap menerima request."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=False)
