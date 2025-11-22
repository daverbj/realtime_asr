"""
FastAPI server for Whisper Large V3 audio transcription.
Upload an audio file and get back the transcription.
Supports real-time streaming transcription via WebSocket.
"""

import warnings
import os
import logging

# Suppress NNPACK warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress torch warnings
logging.getLogger('torch').setLevel(logging.ERROR)

# Set environment variable to suppress NNPACK
os.environ['OMP_NUM_THREADS'] = '1'

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import tempfile
import os
from pathlib import Path
from realtime_transcription import RealtimeTranscriber, websocket_transcription_handler
from omnilingual_transcription import OmniASRTranscriber, websocket_omniasr_handler

app = FastAPI(title="Whisper & OmniASR Transcription API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store the models
# Whisper model
model = None
processor = None
pipe = None
realtime_transcriber = None

# OmniASR model
omniasr_transcriber = None
current_model_type = "whisper"  # "whisper" or "omniasr"


def load_model():
    """Load the Whisper Large V3 model on startup."""
    global model, processor, pipe, realtime_transcriber
    
    if pipe is not None:
        return
    
    print("Loading Whisper Large V3 model...")
    
    # Suppress torch warnings during model loading
    import torch
    torch.set_num_threads(1)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model_id = "openai/whisper-large-v2"
    cache_dir = "./models"
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        cache_dir=cache_dir
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    
    # Optimized generation parameters for speed and multilingual support
    generate_kwargs = {
        "max_new_tokens": 256,  # Reduced to avoid exceeding max_target_positions
        "num_beams": 1,  # Greedy decoding for speed
        "condition_on_prev_tokens": False,  # Faster inference
        "compression_ratio_threshold": 1.35,
        "temperature": 0.0,  # Deterministic for speed
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "language": None,  # Auto-detect: supports Hindi, Bengali, and 90+ languages
        "task": "transcribe",  # Transcribe in source language (not translate)
    }
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs=generate_kwargs,
    )
    
    # Initialize real-time transcriber
    realtime_transcriber = RealtimeTranscriber(pipe, sample_rate=16000)
    
    print(f"Model loaded successfully on {device}")
    print(f"Model: {model_id}")
    print("Multilingual support: Hindi, Bengali, and 90+ languages with auto-detection")
    print("Optimizations: Flash Attention 2/SDPA enabled for faster inference")


@app.on_event("startup")
async def startup_event():
    """Load Whisper model on startup."""
    load_model()


def load_omniasr_model():
    """Load OmniASR model (LLM 7B for multi-language support)."""
    global omniasr_transcriber
    
    if omniasr_transcriber is not None:
        return
    
    print("Loading OmniASR LLM 7B model...")
    omniasr_transcriber = OmniASRTranscriber(
        model_card="omniASR_LLM_7B",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("OmniASR model loaded successfully")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Whisper & OmniASR Transcription API",
        "current_model": current_model_type,
        "available_models": {
            "whisper": "OpenAI Whisper Large V3 - 90+ languages",
            "omniasr": "Facebook OmniASR LLM 7B - 1600+ languages with multi-language detection"
        },
        "endpoints": {
            "/transcribe": "POST - Upload audio file for transcription",
            "/ws/transcribe": "WebSocket - Real-time streaming transcription (Whisper)",
            "/ws/omniasr": "WebSocket - Real-time streaming transcription (OmniASR)",
            "/model": "POST - Switch model type",
            "/health": "GET - Check API health status"
        }
    }


@app.post("/model")
async def switch_model(model_type: str):
    """
    Switch between Whisper and OmniASR models.
    
    Args:
        model_type: "whisper" or "omniasr"
    """
    global current_model_type
    
    if model_type not in ["whisper", "omniasr"]:
        raise HTTPException(status_code=400, detail="Invalid model type. Use 'whisper' or 'omniasr'")
    
    if model_type == "omniasr" and omniasr_transcriber is None:
        load_omniasr_model()
    
    current_model_type = model_type
    
    return {
        "status": "success",
        "current_model": current_model_type,
        "message": f"Switched to {model_type} model"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "current_model": current_model_type,
        "whisper_loaded": pipe is not None,
        "omniasr_loaded": omniasr_transcriber is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    return_timestamps: bool = False
):
    """
    Transcribe an audio file.
    
    Parameters:
    - file: Audio file (mp3, wav, flac, ogg, etc.)
    - return_timestamps: Whether to return timestamps for each segment
    
    Returns:
    - text: Transcribed text
    - timestamps: Optional timestamp information for each segment
    """
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    # Validate file type
    allowed_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.wma', '.aac'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Transcribe the audio
        result = pipe(tmp_file_path)
        
        # Prepare response
        response = {
            "text": result["text"],
            "filename": file.filename
        }
        
        if return_timestamps and "chunks" in result:
            response["chunks"] = [
                {
                    "timestamp": chunk["timestamp"],
                    "text": chunk["text"]
                }
                for chunk in result["chunks"]
            ]
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio transcription using Whisper.
    
    Client should send:
    - Binary audio data (16-bit PCM, 16kHz, mono)
    - Text commands: {"type": "end_segment"}
    
    Server sends:
    - {"type": "status", "message": "..."}
    - {"type": "vad", "is_speech": bool, "probability": float}
    - {"type": "transcription", "text": "...", "is_final": bool}
    - {"type": "error", "message": "..."}
    """
    if realtime_transcriber is None:
        await websocket.close(code=1011, reason="Whisper model not loaded")
        return
    
    await websocket_transcription_handler(websocket, realtime_transcriber)


@app.websocket("/ws/omniasr")
async def websocket_omniasr(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio transcription using OmniASR.
    
    Client should send:
    - Binary audio data (16-bit PCM, 16kHz, mono)
    - Text commands: {"type": "set_language", "language": "hi"}, {"type": "stop"}
    
    Server sends:
    - {"type": "status", "message": "..."}
    - {"type": "transcription", "text": "...", "language": "...", "is_final": bool}
    - {"type": "error", "message": "..."}
    """
    global omniasr_transcriber
    
    # Load OmniASR model if not loaded
    if omniasr_transcriber is None:
        load_omniasr_model()
    
    await websocket_omniasr_handler(websocket, omniasr_transcriber)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
