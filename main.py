from contextlib import asynccontextmanager
from pathlib import Path
import hmac
import os
import re

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import uvicorn

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
except ImportError:
    print("[WARNING] Sastrawi belum terinstall. Stemming tidak akan berjalan.")
    StemmerFactory = None

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
IS_PRODUCTION = APP_ENV == "production"

# Inisialisasi Sastrawi Stemmer
stemmer = None
if StemmerFactory:
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

# Global variables untuk model
tfidf_vectorizer = None
svm_model = None
mlb_classes = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tfidf_vectorizer, svm_model, mlb_classes
    try:
        tfidf_vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer_final.pkl")
        svm_model = joblib.load(MODELS_DIR / "svm_tuned_model.pkl")
        mlb_classes = joblib.load(MODELS_DIR / "mlb_classes_final.pkl")
        print("[AI] Berhasil memuat model SVM, TF-IDF, dan MLB!")
    except Exception as e:
        print(f"[AI ERROR] Gagal memuat model .pkl. Error: {e}")
    yield


app = FastAPI(
    title="MCC AI Microservice",
    description="Microservice untuk prediksi otomatis Program SDGs berbasis NLP (TF-IDF + SVM)",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
)

# Batasi CORS ke origin internal / aplikasi utama
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Internal-Token"],
)


class PredictRequest(BaseModel):
    judul_kegiatan: str = Field(..., min_length=1, max_length=500)
    nama_organisasi: str = Field(default="", max_length=300)


def clean_text(text: str) -> str:
    """Membersihkan dan melakukan stemming pada teks input."""
    if not isinstance(text, str):
        return ""
    text = text[:500].lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if stemmer:
        text = stemmer.stem(text)
    return text


@app.post("/api/predict-sdgs")
def predict_sdgs(
    req: PredictRequest,
    x_internal_token: str | None = Header(default=None),
):
    """
    Menerima Judul Kegiatan dan mengembalikan prediksi kategori SDGs
    menggunakan model SVM yang sudah dilatih.
    """
    expected_token = os.getenv("AI_INTERNAL_TOKEN", "").strip()
    if expected_token:
        if not x_internal_token or not hmac.compare_digest(x_internal_token, expected_token):
            raise HTTPException(status_code=401, detail="Unauthorized internal request.")
    elif IS_PRODUCTION:
        raise HTTPException(status_code=503, detail="Service misconfigured: AI_INTERNAL_TOKEN is required in production.")

    # 1. Pastikan model sudah di-load
    if svm_model is None or tfidf_vectorizer is None or mlb_classes is None:
        return {
            "status": "error",
            "message": "Model AI belum siap atau gagal dimuat.",
            "predicted_sdgs": [],
        }

    # 2. Bersihkan teks (Preprocessing)
    cleaned_text = clean_text(req.judul_kegiatan)

    # 3. TF-IDF Vectorization
    X_input = tfidf_vectorizer.transform([cleaned_text])

    # 4. Prediksi menggunakan SVM
    pred_biner = svm_model.predict(X_input)

    # 5. Inverse Transform (Mengubah array 1 dan 0 kembali menjadi teks SDGS X)
    predicted_labels = mlb_classes.inverse_transform(pred_biner)

    predicted_sdgs = []
    if predicted_labels and len(predicted_labels[0]) > 0:
        predicted_sdgs = list(predicted_labels[0])

    return {
        "status": "success",
        "input": req.judul_kegiatan,
        "predicted_sdgs": predicted_sdgs,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "message": "Microservice AI aktif dan siap menerima request."}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=False)
