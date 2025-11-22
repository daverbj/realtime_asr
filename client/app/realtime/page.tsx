'use client';

import { useState, useRef, useEffect } from 'react';

interface TranscriptionSegment {
  text: string;
  timestamp: number;
  isFinal: boolean;
}

interface VADStatus {
  isSpeech: boolean;
  probability: number;
}

export default function RealtimeTranscription() {
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [transcriptions, setTranscriptions] = useState<TranscriptionSegment[]>([]);
  const [vadStatus, setVadStatus] = useState<VADStatus>({ isSpeech: false, probability: 0 });
  const [status, setStatus] = useState<string>('Not connected');
  const [error, setError] = useState<string>('');
  const [audioDevices, setAudioDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [selectedLanguage, setSelectedLanguage] = useState<string>('auto');
  const [selectedModel, setSelectedModel] = useState<string>('whisper'); // 'whisper' or 'omniasr'

  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    // Get available audio devices
    const getAudioDevices = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = devices.filter((device) => device.kind === 'audioinput');
        setAudioDevices(audioInputs);
        if (audioInputs.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(audioInputs[0].deviceId);
        }
      } catch (err) {
        console.error('Error enumerating devices:', err);
      }
    };

    getAudioDevices();

    // Listen for device changes
    navigator.mediaDevices.addEventListener('devicechange', getAudioDevices);

    return () => {
      // Cleanup on unmount
      navigator.mediaDevices.removeEventListener('devicechange', getAudioDevices);
      if (wsRef.current) {
        wsRef.current.close();
      }
      stopRecording();
    };
  }, []);

  const connectWebSocket = () => {
    // Choose WebSocket endpoint based on selected model
    const endpoint = selectedModel === 'omniasr' 
      ? 'ws://149.36.1.63:28888/ws/omniasr'
      : 'ws://149.36.1.63:28888/ws/transcribe';
    
    const ws = new WebSocket(endpoint);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      setStatus('Connected');
      setError('');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'status':
          setStatus(data.message);
          break;

        case 'vad':
          setVadStatus({
            isSpeech: data.is_speech,
            probability: data.probability,
          });
          break;

        case 'transcription':
          setTranscriptions((prev) => [
            ...prev,
            {
              text: data.text,
              timestamp: Date.now(),
              isFinal: data.is_final,
            },
          ]);
          break;

        case 'error':
          setError(data.message);
          break;
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('WebSocket connection error');
      setIsConnected(false);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      setStatus('Disconnected');
    };

    wsRef.current = ws;
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined,
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      streamRef.current = stream;

      // Create AudioContext for processing
      const audioContext = new AudioContext({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);

      processor.onaudioprocess = (e) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
          return;
        }

        const inputData = e.inputBuffer.getChannelData(0);
        
        // Convert Float32Array to Int16Array
        const int16Data = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }

        // Send to WebSocket
        wsRef.current.send(int16Data.buffer);
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      audioContextRef.current = audioContext;
      processorRef.current = processor;

      // Send language preference to backend
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && selectedLanguage !== 'auto') {
        wsRef.current.send(JSON.stringify({
          type: 'set_language',
          language: selectedLanguage
        }));
      }

      setIsRecording(true);
      setStatus('Recording...');
    } catch (err) {
      console.error('Error accessing microphone:', err);
      setError('Failed to access microphone');
    }
  };

  const stopRecording = () => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    setIsRecording(false);
    setStatus(isConnected ? 'Connected' : 'Disconnected');
  };

  const handleToggleRecording = () => {
    if (!isConnected) {
      connectWebSocket();
      return;
    }

    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const clearTranscriptions = () => {
    setTranscriptions([]);
  };

  const exportTranscript = () => {
    const text = transcriptions.map((t) => t.text).join(' ');
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcript-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold mb-2">Real-Time Transcription</h1>
        <p className="text-gray-400 mb-8">Powered by Whisper Large V3 & OmniASR</p>

        {/* Model Selection */}
        <div className="bg-gray-800 rounded-lg p-6 mb-6 border border-gray-700">
          <h2 className="text-lg font-semibold mb-4">🤖 Model Selection</h2>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">ASR Model</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={isRecording || isConnected}
              className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <option value="whisper">Whisper Large V3 (90+ languages, VAD-based)</option>
              <option value="omniasr">OmniASR CTC 7B (1600+ languages, 16x real-time)</option>
            </select>
            <p className="text-xs text-gray-400 mt-1">
              {selectedModel === 'whisper' 
                ? '🎯 Best for: High accuracy with voice activity detection'
                : '⚡ Best for: Ultra-fast transcription with massive language coverage'}
            </p>
          </div>
        </div>

        {/* Microphone Selection */}
        <div className="bg-gray-800 rounded-lg p-6 mb-6 border border-gray-700">
          <h2 className="text-lg font-semibold mb-4">🎤 Audio Settings</h2>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Microphone</label>
            <select
              value={selectedDeviceId}
              onChange={(e) => setSelectedDeviceId(e.target.value)}
              disabled={isRecording}
              className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {audioDevices.length === 0 ? (
                <option>No microphones detected</option>
              ) : (
                audioDevices.map((device) => (
                  <option key={device.deviceId} value={device.deviceId}>
                    {device.label || `Microphone ${device.deviceId.substring(0, 8)}`}
                  </option>
                ))
              )}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Language</label>
            <select
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="auto">Auto-detect (English, Hindi, Bengali)</option>
              <option value="en">English</option>
              <option value="hi">Hindi (हिंदी)</option>
              <option value="bn">Bengali (বাংলা)</option>
            </select>
            <p className="text-xs text-gray-400 mt-1">
              💡 OmniASR will detect from the selected language(s)
            </p>
          </div>
          
          {isRecording && (
            <p className="text-sm text-gray-400 mt-2">
              You can change language while recording
            </p>
          )}
        </div>

        {/* Status Card */}
        <div className="bg-gray-800 rounded-lg p-6 mb-6 border border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="flex items-center gap-2">
                <div
                  className={`w-3 h-3 rounded-full ${
                    isConnected ? 'bg-green-500' : 'bg-red-500'
                  } animate-pulse`}
                ></div>
                <span className="text-sm font-medium">{status}</span>
              </div>
              {error && <p className="text-red-400 text-sm mt-2">{error}</p>}
            </div>

            <div className="flex gap-2">
              <button
                onClick={handleToggleRecording}
                className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                  isRecording
                    ? 'bg-red-600 hover:bg-red-700'
                    : 'bg-blue-600 hover:bg-blue-700'
                }`}
              >
                {isRecording ? '⏹ Stop' : isConnected ? '🎤 Start Recording' : '🔌 Connect'}
              </button>

              {transcriptions.length > 0 && (
                <>
                  <button
                    onClick={clearTranscriptions}
                    className="px-4 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg"
                  >
                    🗑 Clear
                  </button>
                  <button
                    onClick={exportTranscript}
                    className="px-4 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg"
                  >
                    💾 Export
                  </button>
                </>
              )}
            </div>
          </div>

          {/* VAD Indicator */}
          {isRecording && (
            <div className="mt-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">Voice Activity Detection</span>
                <span className="text-sm font-mono">
                  {(vadStatus.probability * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all ${
                    vadStatus.isSpeech ? 'bg-green-500' : 'bg-gray-600'
                  }`}
                  style={{ width: `${vadStatus.probability * 100}%` }}
                ></div>
              </div>
              <div className="mt-2 text-center">
                <span
                  className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${
                    vadStatus.isSpeech
                      ? 'bg-green-500/20 text-green-400'
                      : 'bg-gray-700 text-gray-400'
                  }`}
                >
                  {vadStatus.isSpeech ? '🎙️ Speaking' : '🤫 Silence'}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Transcription Display */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 min-h-[400px]">
          <h2 className="text-xl font-semibold mb-4">Transcription</h2>
          <div className="space-y-3 max-h-[500px] overflow-y-auto">
            {transcriptions.length === 0 ? (
              <p className="text-gray-500 text-center py-8">
                Start recording to see transcriptions appear here...
              </p>
            ) : (
              transcriptions.map((segment, index) => (
                <div
                  key={index}
                  className="bg-gray-700/50 rounded p-4 border-l-4 border-blue-500 animate-fadeIn"
                >
                  <p className="text-gray-300">{segment.text}</p>
                  <p className="text-xs text-gray-500 mt-2">
                    {new Date(segment.timestamp).toLocaleTimeString()}
                  </p>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Instructions */}
        <div className="mt-6 bg-blue-900/20 border border-blue-700 rounded-lg p-4">
          <h3 className="font-semibold mb-2">📝 How to use:</h3>
          <ul className="text-sm text-gray-300 space-y-1">
            <li>1. Click "Connect" to establish WebSocket connection</li>
            <li>2. Click "Start Recording" to begin transcription</li>
            <li>3. Speak clearly into your microphone</li>
            <li>4. Watch real-time transcriptions appear below</li>
            <li>5. Click "Stop" when finished, then export if needed</li>
          </ul>
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out;
        }
      `}</style>
    </div>
  );
}
