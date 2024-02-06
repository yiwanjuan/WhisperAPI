# ISO_639_1 to language
# Only including those supported by original OpenAI API
ISO_639_1 = {
    "af": "afrikaans",
    "ar": "arabic",
    "hy": "armenian",
    "az": "azerbaijani",
    "be": "belarusian",
    "bs": "bosnian",
    "bg": "bulgarian",
    "ca": "catalan",
    "zh": "chinese",
    "hr": "croatian",
    "cs": "czech",
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "gl": "galician",
    "de": "german",
    "el": "greek",
    "he": "hebrew",
    "hi": "hindi",
    "hu": "hungarian",
    "is": "icelandic",
    "id": "indonesian",
    "it": "italian",
    "ja": "japanese",
    "kn": "kannada",
    "kk": "kazakh",
    "ko": "korean",
    "lv": "latvian",
    "lt": "lithuanian",
    "mk": "macedonian",
    "ms": "malay",
    "mr": "marathi",
    "mi": "maori",
    "ne": "nepali",
    "no": "norwegian",
    "fa": "persian",
    "pl": "polish",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "sr": "serbian",
    "sk": "slovak",
    "sl": "slovenian",
    "es": "spanish",
    "sw": "swahili",
    "sv": "swedish",
    "tl": "tagalog",
    "ta": "tamil",
    "th": "thai",
    "tr": "turkish",
    "uk": "ukrainian",
    "ur": "urdu",
    "vi": "vietnamese",
    "cy": "welsh",
}


# s -> HH:mm:ss.SSS
def timestamp(s: float, format: str = "vtt") -> str:
    m, s = divmod(s, 60)
    H, m = divmod(m, 60)
    HHmmssSSS = "%02d:%02d:%s" % (H, m, str("%.3f" % s).zfill(6))
    if format == "srt":
        HHmmssSSS = HHmmssSSS.replace(".", ",")
    elif format == "vtt":
        pass
    else:
        raise Exception(
            f"'format' should be either 'vtt' or 'srt', but got '{format}'!"
        )
    return HHmmssSSS


def srt_chunk(index: int, start: float, end: float, content: str) -> str:
    content = content.rstrip("\n")
    chunk = f"{index}\n"
    chunk += f'{timestamp(start,"srt")} --> {timestamp(end,"srt")}\n'
    chunk += f"{content}\n\n"
    return chunk


def vtt_chunk(start: float, end: float, content: str) -> str:
    content = content.rstrip("\n")
    chunk = f'{timestamp(start,"vtt")} --> {timestamp(end,"vtt")}\n'
    chunk += f"{content}\n\n"
    return chunk


def whisper2srt(chunks: list[dict]) -> str:
    result = ""
    for index, chunk in enumerate(chunks, start=1):
        if chunk["timestamp"][1] is None:  # Whisper did not predict an ending timestamp
            chunk["timestamp"] = (chunk["timestamp"][0], chunk["timestamp"][0] + 0.001)
        result += srt_chunk(
            index, chunk["timestamp"][0], chunk["timestamp"][1], chunk["text"]
        )
    return result


def whisper2vtt(chunks: list[dict]) -> str:
    result = "WEBVTT\n\n"
    for chunk in chunks:
        if chunk["timestamp"][1] is None:  # Whisper did not predict an ending timestamp
            chunk["timestamp"] = (chunk["timestamp"][0], chunk["timestamp"][0] + 0.001)
        result += vtt_chunk(chunk["timestamp"][0], chunk["timestamp"][1], chunk["text"])
    return result
