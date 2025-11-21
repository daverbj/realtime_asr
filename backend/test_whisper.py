#!/usr/bin/env python3
"""
Test script for OpenAI Whisper Large V3 model from HuggingFace.
This script tests the model's speech-to-text capabilities.
"""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time


def test_whisper_large_v3():
    """Test the Whisper Large V3 model with audio transcription."""
    
    print("=" * 60)
    print("Testing Whisper Large V3 Model")
    print("=" * 60)
    
    # Check for GPU availability
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"\nDevice: {device}")
    print(f"Data type: {torch_dtype}")
    
    model_id = "openai/whisper-large-v3"
    cache_dir = "./models"
    
    print(f"\nLoading model: {model_id}")
    print(f"Cache directory: {cache_dir}")
    start_time = time.time()
    
    # Load model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        cache_dir=cache_dir
    )
    model.to(device)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Create pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    print("\n" + "=" * 60)
    print("Pipeline created successfully!")
    print("=" * 60)
    
    # Test with a sample audio file (you'll need to provide an audio file)
    print("\nTo test transcription, you need an audio file.")
    print("Example usage:")
    print("  result = pipe('path/to/audio.mp3')")
    print("  print(result['text'])")
    
    # Alternatively, test with a sample from HuggingFace datasets
    print("\n" + "=" * 60)
    print("Testing with sample audio from HuggingFace datasets...")
    print("=" * 60)
    
    try:
        from datasets import load_dataset
        
        # Load a sample audio dataset
        dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        sample = dataset[0]["audio"]
        
        print(f"\nSample rate: {sample['sampling_rate']} Hz")
        print("Transcribing...")
        
        start_time = time.time()
        result = pipe(sample)
        transcription_time = time.time() - start_time
        
        print(f"\nTranscription completed in {transcription_time:.2f} seconds")
        print(f"\nTranscribed text: {result['text']}")
        
        if 'chunks' in result:
            print("\nTimestamps:")
            for chunk in result['chunks']:
                print(f"  [{chunk['timestamp'][0]:.2f}s - {chunk['timestamp'][1]:.2f}s]: {chunk['text']}")
    
    except ImportError:
        print("\nNote: Install 'datasets' library to test with sample audio:")
        print("  uv pip install datasets")
    except Exception as e:
        print(f"\nError during transcription test: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_whisper_large_v3()
