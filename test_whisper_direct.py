"""
Direct test of Whisper transcription on tat.wav
"""
import whisper
import json

print("Loading Whisper model...")
model = whisper.load_model('medium')

print("Transcribing uploads/tat.wav...")
result = model.transcribe(
    'uploads/tat.wav',
    language='en',
    temperature=0,  # Deterministic
    best_of=1
)

print("\n" + "="*60)
print("TRANSCRIPTION RESULT:")
print("="*60)
print(f"Text: '{result['text']}'")
print(f"Language: {result['language']}")

if result.get('segments'):
    print(f"\nSegments ({len(result['segments'])} total):")
    for i, seg in enumerate(result['segments']):
        print(f"  [{i}] {seg['start']:.2f}s - {seg['end']:.2f}s: '{seg['text']}'")
        if 'confidence' in seg:
            print(f"       confidence: {seg['confidence']}")

print("\n" + "="*60)
print("FULL JSON:")
print("="*60)
print(json.dumps(result, indent=2))
