"""
Test client for the Whisper transcription API.
"""

import requests

# API endpoint
BASE_URL = "http://localhost:8000"


def test_health():
    """Test the health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(response.json())
    print()


def test_transcribe(audio_file_path: str):
    """Test transcription with an audio file."""
    with open(audio_file_path, 'rb') as f:
        files = {'file': f}
        params = {'return_timestamps': True}
        
        print(f"Transcribing: {audio_file_path}")
        response = requests.post(
            f"{BASE_URL}/transcribe",
            files=files,
            params=params
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nTranscription: {result['text']}\n")
            
            if 'chunks' in result:
                print("Timestamps:")
                for chunk in result['chunks']:
                    print(f"  {chunk['timestamp']}: {chunk['text']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.json())


if __name__ == "__main__":
    # Test health
    test_health()
    
    # Example: test with your audio file
    # test_transcribe("path/to/your/audio.mp3")
    print("To test transcription, call:")
    print("  test_transcribe('path/to/your/audio.mp3')")
