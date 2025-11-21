"""
Real-time audio transcription with WebSocket, Silero VAD, and Whisper.
"""

import asyncio
import json
import numpy as np
import torch
from fastapi import WebSocket, WebSocketDisconnect
from collections import deque
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioBuffer:
    """Buffer to accumulate audio chunks for VAD processing."""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.buffer = deque()
        # Silero VAD requires EXACTLY 512 samples for 16kHz
        self.frame_size = 512  # Fixed size required by Silero VAD
        self.frame_duration_ms = 32  # 512 samples at 16kHz = 32ms
        
    def add_chunk(self, audio_data: bytes):
        """Add audio chunk to buffer."""
        # Convert bytes to float32 numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.buffer.extend(audio_np)
    
    def get_frame(self):
        """Get a frame of audio for VAD processing."""
        if len(self.buffer) >= self.frame_size:
            frame = np.array([self.buffer.popleft() for _ in range(self.frame_size)])
            return torch.from_numpy(frame)
        return None
    
    def get_all(self):
        """Get all buffered audio."""
        if len(self.buffer) > 0:
            audio = np.array(list(self.buffer))
            self.buffer.clear()
            return audio
        return None
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


class SpeechSegmentBuffer:
    """Buffer to accumulate speech segments detected by VAD."""
    
    def __init__(self, sample_rate=16000, min_speech_duration_ms=250):
        self.sample_rate = sample_rate
        self.min_speech_samples = int(sample_rate * min_speech_duration_ms / 1000)
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_frames = 0
        self.max_silence_frames = 30  # ~900ms of silence before ending segment
        
    def add_speech_frame(self, frame: np.ndarray, is_speech: bool):
        """Add frame and track speech state."""
        if is_speech:
            self.is_speaking = True
            self.silence_frames = 0
            self.speech_buffer.append(frame)
        elif self.is_speaking:
            self.silence_frames += 1
            self.speech_buffer.append(frame)
            
            # End segment after sustained silence
            if self.silence_frames >= self.max_silence_frames:
                return self._finalize_segment()
        
        return None
    
    def _finalize_segment(self):
        """Finalize and return speech segment."""
        if len(self.speech_buffer) > 0:
            segment = np.concatenate(self.speech_buffer)
            # Only return if segment is long enough
            if len(segment) >= self.min_speech_samples:
                self.speech_buffer = []
                self.is_speaking = False
                self.silence_frames = 0
                return segment
        
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_frames = 0
        return None
    
    def force_finalize(self):
        """Force finalize current segment."""
        return self._finalize_segment()


class RealtimeTranscriber:
    """Real-time transcription with VAD and Whisper."""
    
    def __init__(self, whisper_pipe, sample_rate=16000):
        self.pipe = whisper_pipe
        self.sample_rate = sample_rate
        
        # Load Silero VAD model
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        
        self.get_speech_timestamps = utils[0]
        self.vad_threshold = 0.5
        
        print("Silero VAD model loaded")
    
    def detect_speech(self, audio_tensor: torch.Tensor) -> float:
        """Detect speech in audio frame using VAD."""
        # Silero VAD requires EXACTLY 512 samples for 16kHz
        if len(audio_tensor) != 512:
            return 0.0
        
        speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
        return speech_prob
    
    def transcribe_segment(self, audio_segment: np.ndarray, language: str = None) -> dict:
        """Transcribe an audio segment using Whisper."""
        try:
            # Whisper expects float32 in range [-1, 1]
            # Language auto-detection for multilingual support (Hindi, Bengali, etc.)
            generate_kwargs = {
                "max_new_tokens": 128,  # Conservative limit for short segments
                "task": "transcribe",  # Transcribe in original language
            }
            
            # Force language if specified (not auto)
            if language and language != "auto":
                generate_kwargs["language"] = language
                logger.info(f"Forcing language: {language}")
            
            result = self.pipe(
                audio_segment.astype(np.float32),
                generate_kwargs=generate_kwargs,
                return_timestamps=False,
            )
            
            # Get detected or forced language
            detected_lang = language if language else "auto"
            
            return {
                "text": result["text"].strip(),
                "language": detected_lang
            }
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {"text": "", "language": "error"}


async def websocket_transcription_handler(
    websocket: WebSocket,
    transcriber: RealtimeTranscriber
):
    """Handle WebSocket connection for real-time transcription."""
    await websocket.accept()
    
    logger.info("WebSocket connection accepted")
    
    audio_buffer = AudioBuffer(sample_rate=16000)
    speech_buffer = SpeechSegmentBuffer(sample_rate=16000)
    language_preference = None  # Store language preference
    
    try:
        await websocket.send_json({
            "type": "status",
            "message": "Connected. Start speaking..."
        })
        
        logger.info("Sent connection status to client")
        
        while True:
            # Receive audio data
            data = await websocket.receive()
            
            if "bytes" in data:
                # Add audio chunk to buffer
                audio_buffer.add_chunk(data["bytes"])
                logger.debug(f"Received audio chunk: {len(data['bytes'])} bytes")
                
                # Process frames with VAD
                while True:
                    frame = audio_buffer.get_frame()
                    if frame is None:
                        break
                    
                    # Detect speech
                    speech_prob = transcriber.detect_speech(frame)
                    is_speech = speech_prob > transcriber.vad_threshold
                    
                    # Send VAD status
                    await websocket.send_json({
                        "type": "vad",
                        "is_speech": is_speech,
                        "probability": float(speech_prob)
                    })
                    
                    # Accumulate speech segment
                    frame_np = frame.numpy()
                    segment = speech_buffer.add_speech_frame(frame_np, is_speech)
                    
                    if segment is not None:
                        # Transcribe the speech segment
                        result = transcriber.transcribe_segment(segment)
                        
                        if result["text"]:
                            await websocket.send_json({
                                "type": "transcription",
                                "text": result["text"],
                                "language": result["language"],
                                "is_final": True
                            })
            
            elif "text" in data:
                # Handle text commands
                message = json.loads(data["text"])
                
                if message.get("type") == "end_segment":
                    # Force finalize current segment
                    segment = speech_buffer.force_finalize()
                    if segment is not None:
                        language = message.get("language")  # Optional language hint
                        result = transcriber.transcribe_segment(segment, language)
                        if result["text"]:
                            await websocket.send_json({
                                "type": "transcription",
                                "text": result["text"],
                                "language": result["language"],
                                "is_final": True
                            })
                
                elif message.get("type") == "set_language":
                    # Store language preference for next segments
                    language_preference = message.get("language", "auto")
                    logger.info(f"Language preference set to: {language_preference}")
    
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
