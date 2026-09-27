
from faster_whisper import WhisperModel

# Load the model once when the file is imported
# "small" = small model (fast, good enough for Hindi/English)
# "int8" = compressed version (uses less RAM, faster on CPU)
# cpu_threads=2 = don't overload your RAM (yours is at 92%)
model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=2)


def detect_language(audio_path: str) -> str:
    """
    Listen to the audio and detect what language it is.
    Returns "hi" for Hindi, "en" for English, etc.
    """
    _, info = model.transcribe(audio_path, beam_size=1)
    print(f"  Detected language: {info.language} (confidence: {info.language_probability:.2f})")
    return info.language


def transcribe_all(chunks: list, translate: bool = True) -> str:
    """
    Takes a list of audio chunks and converts speech to text.

    chunks    = list of audio file paths (e.g. ["chunk_0.wav", "chunk_1.wav"])
    translate = True  → always give output in English (even if audio is Hindi/Hinglish)
                False → give output in original language
    
    NOTE: language is auto detected per chunk
          so Hinglish videos are handled automatically
    """

    full_text = ""  # we will keep adding text here as each chunk is done

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}/{len(chunks)}")

        # detect language for EACH chunk automatically
        # this handles Hinglish — some chunks may be "hi", some "en"
        lang = detect_language(chunk)

        # Hinglish handling:
        # if confidence is low it's probably Hinglish (mixed Hindi+English)
        # in that case just let whisper figure it out without forcing a language
        _, info = model.transcribe(chunk, beam_size=1)
        if info.language_probability < 0.8:
            print(f"  Low confidence — likely Hinglish, auto-detecting...")
            lang = None   # None = whisper decides on its own per segment

        # task = "translate" means output will always be in English
        # task = "transcribe" means output stays in original language
        if lang == "en":
            task = "transcribe"
        else:
            task = "translate"

        print(f"  Language: {lang or 'auto'} | Task: {task}")

        # transcribe the chunk
        segments, _ = model.transcribe(
            chunk,
            language=lang,   # None for Hinglish = auto per segment
            task=task,
            beam_size=1,     # beam_size=1 = fastest
            vad_filter=True, # skip silent parts automatically
        )

        # print each sentence live as it's transcribed
        for s in segments:
            full_text += s.text.strip() + " "

    print("\nTranscription complete.")

    print("\n" + "=" * 60)
    print("Transcript:\n")
    print(full_text.strip())
    print("=" * 60)

    return full_text.strip()