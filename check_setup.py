from gnani.tts import GnaniTTSClient

from src.llm import chat
from tools import lookup_order

print(chat("You are ZentroMart's AI support assistant. Reply in one short sentence.", [{"role": "user", "content": "Hi"}]))
print(lookup_order("8978 2837"))
print(lookup_order("8978 2837", "4533")["status"])

'''GnaniTTSClient().synthesize(
    "Hello, this is ZentroMart's AI assistant.",
    voice="Kaveri",
    model="timbre-v2.5",
    language="en-IN",
    output_file="test.wav",
)
print("test.wav written") '''