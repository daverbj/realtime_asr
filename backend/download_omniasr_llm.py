"""
Download OmniASR LLM 7B model with progress bar.
"""
import os
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# Set cache directory
cache_dir = os.path.abspath("./models")
os.makedirs(cache_dir, exist_ok=True)
os.environ['FAIRSEQ2_CACHE_DIR'] = cache_dir

print(f"Downloading OmniASR LLM 7B model...")
print(f"Cache directory: {cache_dir}")
print("This may take a while depending on your internet connection...")

# Initialize pipeline - this will trigger the download
pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B")

print("\n✅ Download complete!")
print(f"Model saved to: {cache_dir}")
