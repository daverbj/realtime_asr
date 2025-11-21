"""
Real-time transcription using OmniASR models from Facebook Research.
Supports 1600+ languages with state-of-the-art accuracy.
"""

import asyncio
import logging
import numpy as np
import torch
from typing import Optional, Dict, Any
from fastapi import WebSocket
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OmniASRBuffer:
    """Buffer for accumulating audio before transcription."""
    
    def __init__(self, sample_rate=16000, chunk_duration=2.0):
        """
        Initialize audio buffer.
        
        Args:
            sample_rate: Audio sample rate (default 16kHz)
            chunk_duration: Duration in seconds before transcribing (default 2s)
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.buffer = []
        self.total_samples = 0
        
    def add_audio(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Add audio chunk to buffer.
        
        Returns segment when buffer reaches chunk_duration, None otherwise.
        """
        self.buffer.append(audio_chunk)
        self.total_samples += len(audio_chunk)
        
        # Return segment when we have enough audio
        if self.total_samples >= self.chunk_samples:
            segment = np.concatenate(self.buffer)
            # Keep overflow for next segment
            if self.total_samples > self.chunk_samples:
                overflow = segment[self.chunk_samples:]
                segment = segment[:self.chunk_samples]
                self.buffer = [overflow]
                self.total_samples = len(overflow)
            else:
                self.buffer = []
                self.total_samples = 0
            return segment
        
        return None
    
    def force_finalize(self) -> Optional[np.ndarray]:
        """Force return current buffer contents."""
        if len(self.buffer) > 0 and self.total_samples > 0:
            segment = np.concatenate(self.buffer)
            self.buffer = []
            self.total_samples = 0
            return segment
        return None


class OmniASRTranscriber:
    """Real-time transcription using OmniASR models."""
    
    def __init__(self, model_card="omniASR_CTC_7B", device="cuda", cache_dir="./models"):
        """
        Initialize OmniASR transcriber.
        
        Args:
            model_card: Model to use (omniASR_CTC_7B recommended for speed)
            device: Device to run on (cuda/cpu)
            cache_dir: Directory to cache downloaded models
        """
        self.model_card = model_card
        self.device = device
        self.sample_rate = 16000
        
        # Set cache directory for fairseq2
        import os
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.abspath(cache_dir)
        os.environ['FAIRSEQ2_CACHE_DIR'] = cache_path
        
        logger.info(f"Loading OmniASR model: {model_card}")
        logger.info(f"Cache directory: {cache_path}")
        logger.info(f"Supported languages: {len(supported_langs)}")
        
        # Initialize pipeline
        self.pipeline = ASRInferencePipeline(model_card=model_card)
        
        # Load Silero VAD model for speech detection
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        self.get_speech_timestamps = utils[0]
        self.vad_threshold = 0.5
        
        logger.info(f"OmniASR model loaded successfully")
        logger.info(f"Model: {model_card}")
        logger.info(f"Device: {device}")
        logger.info("Silero VAD loaded for speech detection")
        
    def get_language_code(self, language: str) -> str:
        """
        Convert language preference to OmniASR language code.
        
        OmniASR uses format: {language_code}_{script}
        Examples: eng_Latn, hin_Deva, ben_Beng, urd_Arab
        """
        # Mapping from common language codes to OmniASR format
        lang_map = {
            "en": "eng_Latn",
            "hi": "hin_Deva",  # Hindi in Devanagari script
            "bn": "ben_Beng",  # Bengali in Bengali script
            "ur": "urd_Arab",  # Urdu in Arabic script
            "pa": "pan_Guru",  # Punjabi in Gurmukhi script
            "ta": "tam_Taml",  # Tamil
            "te": "tel_Telu",  # Telugu
            "mr": "mar_Deva",  # Marathi in Devanagari script
            "gu": "guj_Gujr",  # Gujarati
            "kn": "kan_Knda",  # Kannada
            "ml": "mal_Mlym",  # Malayalam
            "or": "ory_Orya",  # Odia
            "as": "asm_Beng",  # Assamese in Bengali script
            "auto": None,      # Auto-detect (not supported, will use eng_Latn)
        }
        
        code = lang_map.get(language, "eng_Latn")
        
        # Verify language is supported
        if code and code not in supported_langs:
            logger.warning(f"Language code {code} not in supported languages, using eng_Latn")
            code = "eng_Latn"
        
        return code
    
    def transcribe_segment(self, audio_segment: np.ndarray, language: str = "auto") -> Dict[str, Any]:
        """
        Transcribe an audio segment.
        
        Args:
            audio_segment: Audio data as float32 array
            language: Language code (e.g., "hi", "bn", "en")
        
        Returns:
            Dictionary with 'text' and 'language'
        """
        try:
            # Check if there's actual speech using VAD
            audio_tensor = torch.from_numpy(audio_segment).float()
            
            # Get speech probability for the entire segment
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                sampling_rate=self.sample_rate,
                threshold=self.vad_threshold,
                min_speech_duration_ms=250,
                min_silence_duration_ms=100
            )
            
            # If no speech detected, return empty
            if not speech_timestamps:
                logger.info("No speech detected in segment")
                return {
                    "text": "",
                    "language": "no_speech"
                }
            
            # Convert language code
            lang_code = self.get_language_code(language)
            if lang_code is None:
                lang_code = "eng_Latn"  # Default to English
            
            logger.info(f"Speech detected, transcribing with language: {lang_code}")
            
            # Prepare audio data in OmniASR format
            audio_data = {
                "waveform": audio_segment.astype(np.float32),
                "sample_rate": self.sample_rate
            }
            
            # Transcribe
            # OmniASR expects list of audio files or audio dictionaries
            transcriptions = self.pipeline.transcribe(
                [audio_data],
                lang=[lang_code],
                batch_size=1
            )
            
            text = transcriptions[0] if transcriptions else ""
            
            return {
                "text": text.strip(),
                "language": lang_code
            }
            
        except Exception as e:
            logger.error(f"OmniASR transcription error: {e}", exc_info=True)
            return {
                "text": "",
                "language": "error"
            }


async def websocket_omniasr_handler(
    websocket: WebSocket,
    transcriber: OmniASRTranscriber
):
    """Handle WebSocket connection for OmniASR real-time transcription."""
    await websocket.accept()
    
    logger.info("WebSocket connection accepted (OmniASR)")
    
    audio_buffer = OmniASRBuffer(sample_rate=16000, chunk_duration=2.0)
    language_preference = "auto"
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive()
            
            if "bytes" in data:
                # Audio data received
                audio_bytes = data["bytes"]
                
                # Convert bytes to numpy array (assuming 16-bit PCM)
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                
                # Convert to float32 in range [-1, 1]
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0
                
                # Add to buffer
                segment = audio_buffer.add_audio(audio_chunk)
                
                # Transcribe when we have a complete segment
                if segment is not None:
                    logger.info(f"Transcribing segment of {len(segment)} samples")
                    
                    # Transcribe in background to not block receiving audio
                    result = await asyncio.to_thread(
                        transcriber.transcribe_segment,
                        segment,
                        language_preference
                    )
                    
                    if result["text"]:
                        await websocket.send_json({
                            "type": "transcription",
                            "text": result["text"],
                            "language": result["language"],
                            "is_final": True
                        })
                        logger.info(f"Transcription: {result['text']}")
            
            elif "text" in data:
                # Control message
                message = await websocket.receive_json()
                
                if message.get("type") == "set_language":
                    language_preference = message.get("language", "auto")
                    logger.info(f"Language preference set to: {language_preference}")
                
                elif message.get("type") == "stop":
                    # Finalize remaining audio
                    final_segment = audio_buffer.force_finalize()
                    if final_segment is not None and len(final_segment) > 0:
                        result = await asyncio.to_thread(
                            transcriber.transcribe_segment,
                            final_segment,
                            language_preference
                        )
                        if result["text"]:
                            await websocket.send_json({
                                "type": "transcription",
                                "text": result["text"],
                                "language": result["language"],
                                "is_final": True
                            })
                    break
    
    except Exception as e:
        logger.error(f"WebSocket error (OmniASR): {e}", exc_info=True)
    
    finally:
        logger.info("WebSocket connection closed (OmniASR)")
