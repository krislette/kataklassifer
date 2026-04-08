"""
Shared constants used across the gairaigo origin classifier pipeline.

ISO_639_2_NAMES maps the three-letter ISO 639-2 language codes that JMdict
uses in its <lsource> tags to their full English language names. This is used
to replace raw codes like 'fre' or 'ger' with readable labels like 'French'
or 'German' throughout the pipeline — in charts, printed reports, and CSVs.

If a code appears in the data but is not listed here, the code itself is kept
as-is so nothing breaks silently. New codes can simply be added to the dict.

Reference: https://www.loc.gov/standards/iso639-2/php/code_list.php
"""

ISO_639_2_NAMES: dict[str, str] = {
    # Most common donor languages in JMdict
    "eng": "English",
    "fre": "French",
    "ger": "German",
    "por": "Portuguese",
    "dut": "Dutch",
    "ita": "Italian",
    "spa": "Spanish",
    "chi": "Chinese",
    "kor": "Korean",
    "rus": "Russian",
    "ara": "Arabic",
    "lat": "Latin",
    "grc": "Ancient Greek",
    "gre": "Modern Greek",
    "san": "Sanskrit",
    "ain": "Ainu",
    # Less common but present in JMdict
    "afr": "Afrikaans",
    "alb": "Albanian",
    "arm": "Armenian",
    "bnt": "Bantu",
    "bur": "Burmese",
    "cze": "Czech",
    "dan": "Danish",
    "egy": "Ancient Egyptian",
    "epo": "Esperanto",
    "fin": "Finnish",
    "geo": "Georgian",
    "haw": "Hawaiian",
    "heb": "Hebrew",
    "hin": "Hindi",
    "hun": "Hungarian",
    "ice": "Icelandic",
    "ind": "Indonesian",
    "iri": "Irish",
    "khm": "Khmer",
    "may": "Malay",
    "mol": "Moldavian",
    "mon": "Mongolian",
    "nor": "Norwegian",
    "per": "Persian",
    "pol": "Polish",
    "rum": "Romanian",
    "scr": "Croatian",
    "slo": "Slovak",
    "slv": "Slovenian",
    "swa": "Swahili",
    "swe": "Swedish",
    "tha": "Thai",
    "tib": "Tibetan",
    "tur": "Turkish",
    "ukr": "Ukrainian",
    "vie": "Vietnamese",
    "wel": "Welsh",
    "yid": "Yiddish",
    # Catch-all label for consolidated rare classes
    "other": "Other",
}


def decode_language(code: str) -> str:
    """
    Convert an ISO 639-2 code to a full language name.

    Falls back to the code itself if it is not in the mapping table,
    so the pipeline never crashes on an unexpected code.

    Args:
        code : ISO 639-2 three-letter language code (e.g. 'fre').

    Returns:
        Full language name (e.g. 'French'), or the original code if unknown.
    """
    return ISO_639_2_NAMES.get(code, code)
