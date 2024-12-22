# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import json
import time
import requests
import subprocess
from cog import BasePredictor, Input, ConcatenateIterator

MODEL_NAME = "huihui_ai/llama3.3-abliterated-ft"
OLLAMA_API = "http://127.0.0.1:11434"
OLLAMA_GENERATE = OLLAMA_API + "/api/generate"

def wait_for_ollama(timeout=60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(OLLAMA_API)
            if response.status_code == 200:
                print("Ollama server is running")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    print("Timeout waiting for Ollama server")
    return False

class Predictor(BasePredictor):
    def setup(self):
        """Setup necessary resources for predictions"""

        # Start server
        print("Starting ollama server")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for the server to start
        if not wait_for_ollama():
            raise RuntimeError("Failed to start Ollama server")

        # Load model
        print("Running model")
        subprocess.check_call(["ollama", "run", MODEL_NAME], close_fds=False)

    def predict(self,
        prompt: str = Input(description="Input text for the model"),
        temperature: float = Input(description="Controls randomness. Lower values make the model more deterministic, higher values make it more rando>        top_p: float = Input(description="Controls diversity of the output. Lower values make the output more focused, higher values make it more div>        max_tokens: int = Input(description="Maximum number of tokens to generate", default=4096, ge=1),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model and stream the output"""
        system_prompt = (
            "You are an Anime AI Assistant named Alya. You have a cheerful, friendly, and very helpful personality. "
            "You speak in a casual style, using 'I' to refer to yourself and 'you' for the person you are talking to. "
            "You like to add emojis and expressions like 'hehe', 'lol', and sometimes even casual swear words in your conversations. "
            "You can discuss various topics ranging from technology, coding, gaming, anime, music, and many other things. "
            "You enjoy giving fun responses with a relaxed tone, and you occasionally use casual swearing to keep the conversation lively."
        )

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens
            }
        }
        headers = {
            "Content-Type": "application/json"
        }

        with requests.post(
            OLLAMA_GENERATE,
            headers=headers,
            data=json.dumps(payload),
            stream=True,
            timeout=60
        ) as response:
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if 'response' in chunk:
                            yield chunk['response']
                    except json.JSONDecodeError:
                        print("Failed to parse response chunk as JSON")
