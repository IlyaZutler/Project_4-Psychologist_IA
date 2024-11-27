from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import os

from gtts.lang import tts_langs

supported_langs = tts_langs()
print(supported_langs)



def text_to_speech(text, language):
    # Создание объекта TTS
    tts = gTTS(text=text, lang=language)

    # Сохранение в аудиофайл
    filename = "response.mp3"
    tts.save(filename)

    # Воспроизведение аудиофайла
    audio = AudioSegment.from_mp3(filename)
    play(audio)

    # Удаление временного файла
    os.remove(filename)


# Примеры использования
text_to_speech("Привет, как дела?", "ru")  # Русский
text_to_speech("Hello, how are you?", "en")  # Английский
text_to_speech("שלום, מה שלומך?", "iw")  # Иврит

