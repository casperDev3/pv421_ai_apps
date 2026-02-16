import os  # бібліотека для роботи з операційною системою
import requests  # бібліотека для роботи з HTTP-запитами
import speech_recognition as sr  # бібліотека для розпізнавання мови
from gtts import gTTS  # бібліотека для перетворення тексту в мову
import tempfile  # бібліотека для створення тимчасових файлів
import platform  # бібліотека для визначення операційної системи
import subprocess  # бібліотека для виконання системних команд
from pygame import mixer  # бібліотека для відтворення аудіо (необов'язково, можна використовувати системні засоби)

# --- Конфіг ---
GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    raise RuntimeError("❌ Установіть GROQ_API_KEY (https://console.groq.com/keys)")

MODEL = "llama-3.3-70b-versatile"  # або "llama-3.1-8b-instant"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def play_audio(path: str):
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(path)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", path])
        else:  # Linux та інші
            subprocess.run(["xdg-open", path])
    except Exception as e:
        print(f"❌ Не вдалося відтворити аудіо: {e}")


def listen_ukrainian(timeout=5, phrase_time_limit=20):
    r = sr.Recognizer()
    # 🎤 Тут створюється об’єкт мікрофона (можна змінити device_index)
    with sr.Microphone(device_index=5) as source:
        print("🎤 Говори... (українською)")
        print(sr.Microphone.list_microphone_names())
        r.adjust_for_ambient_noise(source, duration=0.8)
        try:
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            print("⏱️ Не почув нічого.")
            return None
    try:
        text = r.recognize_google(audio, language="uk-UA")
        print("👂 Ти сказав:", text)
        return text
    except Exception:
        print("⚠️ Не вдалося розпізнати.")
        return None

def ask_groq(prompt, system="You are a helpful assistant that answers in Ukrainian."):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1000,
        "stream": False
    }

    try:
        response = requests.post(GROQ_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"❌ Помилка при запиті до GROQ: {e}")
        return "Вибач, сталася помилка при отриманні відповіді."

def speak_ua(text):
    try:
        tts = gTTS(text=text, lang='uk')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            # play_audio(tmp_file.name)
            # os.remove(tmp_file.name)  # видаляємо тимчасовий файл після відтворення
        mixer.init()
        mixer.music.load(tmp_file.name)
        mixer.music.play()
        while mixer.music.get_busy():
            continue
        os.remove(tmp_file.name)  # видаляємо тимчасовий файл після від
    except Exception as e:
        print(f"❌ Помилка при перетворенні тексту в мову: {e}")


def main():
    while True:
        query = listen_ukrainian()
        if not query:
            continue

        if query.lower() in ["вихід", "стоп", "завершити"]:
            print("👋 До побачення!")
            speak_ua("До побачення!")
            break

        print("Думаю...")
        answer = ask_groq(query)
        print("🤖 Відповідь:", answer)
        speak_ua(answer)


if __name__ == "__main__":
    main()
