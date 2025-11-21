# Whisper Large V3 Testing

This directory contains a test script for the OpenAI Whisper Large V3 model from HuggingFace.

## Setup

1. Activate the virtual environment:
```bash
source .venv/bin/activate
```

2. Dependencies are already installed via `requirements.txt`:
   - torch
   - transformers
   - accelerate
   - datasets
   - soundfile
   - librosa

## Running the Test

Run the test script:
```bash
python test_whisper.py
```

The script will:
1. Load the Whisper Large V3 model from HuggingFace
2. Create an automatic speech recognition pipeline
3. Test transcription with a sample audio file from HuggingFace datasets

## Model Information

- **Model**: openai/whisper-large-v3
- **HuggingFace URL**: https://huggingface.co/openai/whisper-large-v3
- **Task**: Automatic Speech Recognition (ASR)
- **Size**: ~3GB (large model)

## Notes

- The model will be downloaded to your HuggingFace cache (~/.cache/huggingface/) on first run
- GPU acceleration is used if available (CUDA)
- The test uses a sample from the LibriSpeech ASR dataset for validation

## Testing with Your Own Audio

To test with your own audio file, modify the script:

```python
result = pipe('path/to/your/audio.mp3')
print(result['text'])
```

Supported formats: WAV, MP3, FLAC, OGG, etc.
