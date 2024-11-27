import pyttsx3


def text_to_speech(text, language):
    """
    Озвучивает текст на выбранном языке: русском, английском или иврите.
    Если язык не поддерживается, используется английский по умолчанию.

    :param text: Строка текста для озвучивания.
    :param language: Код языка ('ru', 'en', 'he').
    """
    # Инициализация pyttsx3
    engine = pyttsx3.init()

    # Установим голос на основе языка
    voices = engine.getProperty('voices')
    if language == 'ru':  # Русский
        voice_id = next((voice.id for voice in voices if 'russian' in voice.name.lower()), None)
    elif language == 'he':  # Иврит
        voice_id = next((voice.id for voice in voices if 'hebrew' in voice.name.lower()), None)
    else:  # Английский по умолчанию
        voice_id = next((voice.id for voice in voices if 'english' in voice.name.lower()), None)

    if voice_id:
        engine.setProperty('voice', voice_id)
    else:
        print(f"Язык '{language}' не поддерживается. Используется английский по умолчанию.")
        engine.setProperty('voice', next((voice.id for voice in voices if 'english' in voice.name.lower()), None))

    # Установим скорость речи
    engine.setProperty('rate', 150)  # Скорость речи (уменьшить для большей разборчивости)

    # Воспроизведение текста
    engine.say(text)
    engine.runAndWait()

text_to_speech("Привет, как дела?", "ru")  # Русский
text_to_speech("Hello, how are you?", "en")  # Английский
text_to_speech("שלום, מה שלומך?", "he")  # Иврит


engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    print(f"Voice: {voice.name}, ID: {voice.id}, Languages: {voice.languages}")

