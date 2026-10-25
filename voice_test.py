from src.speech import speak

samples = {
    "en": "Hello, I am ZentroMart's AI assistant. Your refund should arrive by 27 October 2026.",
    "hi": "नमस्ते, मैं ज़ेन्ट्रोमार्ट का AI सहायक हूँ। आपका रिफंड 27 अक्टूबर 2026 तक आ जाएगा।",
    "hi-en": "Namaste, main ZentroMart ka AI assistant hoon. Aapka refund 27 October 2026 tak aa jayega.",
}

for language, text in samples.items():
    print(speak(text, language, f"voice_samples/{language}.wav"))