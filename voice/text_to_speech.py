from gtts import gTTS
import os

def text_to_speech(text, output_file='output.mp3'):
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)
    print(f"Speech saved to {output_file}")

if __name__ == "__main__":
    text = "Hello, this is a test of the gTTS text-to-speech API."
    text_to_speech(text)
