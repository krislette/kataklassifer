"""
Parses the full JMdict XML and exports all gairaigo (外来語) entries
grouped by donor language, with country metadata for the world map.

Output: gairaigo_full.json
  {
    "eng": {
      "language": "English",
      "country": "United Kingdom",
      "iso2": "GB",
      "words": [
        { "katakana": "コーヒー", "meaning": "coffee" },
        ...
      ]
    },
    ...
  }

Usage:
    python export_gairaigo.py --jmdict data/JMdict --out data/gairaigo_full.json
"""

import argparse
import json
import re
from lxml import etree
from pathlib import Path


# ISO 639-2 → { language name, country name, ISO 3166-1 alpha-2 } mapping
# Only languages actually present in JMdict as lsource donor codes.
# "iso2" is the country code used by the D3 world map (TopoJSON).
LANGUAGE_META: dict[str, dict] = {
    # Core European donors
    "eng": {"language": "English", "country": "United Kingdom", "iso2": "GB"},
    "fre": {"language": "French", "country": "France", "iso2": "FR"},
    "ger": {"language": "German", "country": "Germany", "iso2": "DE"},
    "por": {"language": "Portuguese", "country": "Portugal", "iso2": "PT"},
    "spa": {"language": "Spanish", "country": "Spain", "iso2": "ES"},
    "ita": {"language": "Italian", "country": "Italy", "iso2": "IT"},
    "dut": {"language": "Dutch", "country": "Netherlands", "iso2": "NL"},
    "rus": {"language": "Russian", "country": "Russia", "iso2": "RU"},
    "swe": {"language": "Swedish", "country": "Sweden", "iso2": "SE"},
    "nor": {"language": "Norwegian", "country": "Norway", "iso2": "NO"},
    "dan": {"language": "Danish", "country": "Denmark", "iso2": "DK"},
    "fin": {"language": "Finnish", "country": "Finland", "iso2": "FI"},
    "gre": {"language": "Greek", "country": "Greece", "iso2": "GR"},
    "pol": {"language": "Polish", "country": "Poland", "iso2": "PL"},
    "cze": {"language": "Czech", "country": "Czech Republic", "iso2": "CZ"},
    "hun": {"language": "Hungarian", "country": "Hungary", "iso2": "HU"},
    "rum": {"language": "Romanian", "country": "Romania", "iso2": "RO"},
    # Asian donors
    "chi": {"language": "Chinese", "country": "China", "iso2": "CN"},
    "kor": {"language": "Korean", "country": "South Korea", "iso2": "KR"},
    "vie": {"language": "Vietnamese", "country": "Vietnam", "iso2": "VN"},
    "mal": {"language": "Malay", "country": "Malaysia", "iso2": "MY"},
    "tgl": {"language": "Tagalog", "country": "Philippines", "iso2": "PH"},
    "ind": {"language": "Indonesian", "country": "Indonesia", "iso2": "ID"},
    "tha": {"language": "Thai", "country": "Thailand", "iso2": "TH"},
    "bur": {"language": "Burmese", "country": "Myanmar", "iso2": "MM"},
    "khm": {"language": "Khmer", "country": "Cambodia", "iso2": "KH"},
    # South Asian donors
    "san": {"language": "Sanskrit", "country": "India", "iso2": "IN"},
    "hin": {"language": "Hindi", "country": "India", "iso2": "IN"},
    "tam": {"language": "Tamil", "country": "India", "iso2": "IN"},
    "urd": {"language": "Urdu", "country": "Pakistan", "iso2": "PK"},
    "ben": {"language": "Bengali", "country": "Bangladesh", "iso2": "BD"},
    # Middle Eastern / Central Asian donors
    "ara": {"language": "Arabic", "country": "Saudi Arabia", "iso2": "SA"},
    "per": {"language": "Persian", "country": "Iran", "iso2": "IR"},
    "tur": {"language": "Turkish", "country": "Turkey", "iso2": "TR"},
    # African donors
    "swa": {"language": "Swahili", "country": "Tanzania", "iso2": "TZ"},
    "amh": {"language": "Amharic", "country": "Ethiopia", "iso2": "ET"},
    # Americas donors
    "ain": {"language": "Ainu", "country": "Japan", "iso2": "JP"},
    "grn": {"language": "Guaraní", "country": "Paraguay", "iso2": "PY"},
    "que": {"language": "Quechua", "country": "Peru", "iso2": "PE"},
    "nah": {"language": "Nahuatl", "country": "Mexico", "iso2": "MX"},
    # Classical / ecclesiastical
    "lat": {"language": "Latin", "country": "Italy", "iso2": "IT"},
    "heb": {"language": "Hebrew", "country": "Israel", "iso2": "IL"},
    # Oceanian donors
    "mao": {"language": "Māori", "country": "New Zealand", "iso2": "NZ"},
    # Catch-all fallback (JMdict default: absent xml:lang = English)
    "unknown": {"language": "Unknown", "country": "Unknown", "iso2": "XX"},
}

# Katakana Unicode block: \u30A0–\u30FF  (+ prolonged sound mark \u30FC)
KATAKANA_RE = re.compile(r"^[\u30A0-\u30FF\u30FC\u30FB\u30FE\u30FD]+$")


def is_katakana(text: str) -> bool:
    return bool(KATAKANA_RE.match(text))


def parse_jmdict(jmdict_path: Path) -> dict[str, dict]:
    """
    Parse JMdict XML and return data grouped by ISO 639-2 donor language code.
    Returns:
        {
            "eng": { ...meta..., "words": [{"katakana": ..., "meaning": ...}, ...] },
            ...
        }
    """
    result: dict[str, dict] = {}
    seen: set[tuple[str, str]] = set()  # (katakana, lang_code) dedup

    context = etree.iterparse(str(jmdict_path), events=("end",), tag="entry")

    for _, entry in context:
        # Collect katakana readings
        katakana_forms: list[str] = []

        # Prefer k_ele (kanji element) that is pure katakana
        for k_ele in entry.findall("k_ele/keb"):
            if k_ele.text and is_katakana(k_ele.text):
                katakana_forms.append(k_ele.text)

        # Fall back to r_ele (reading element)
        if not katakana_forms:
            for r_ele in entry.findall("r_ele/reb"):
                if r_ele.text and is_katakana(r_ele.text):
                    katakana_forms.append(r_ele.text)

        if not katakana_forms:
            entry.clear()
            continue

        # Extra sense blocks
        for sense in entry.findall("sense"):
            lsource = sense.find("lsource")
            if lsource is None:
                entry.clear()
                break  # no lsource, means not a gairaigo entry

            # JMdict convention: absent xml:lang defaults to English
            lang_code = lsource.get("{http://www.w3.org/XML/1998/namespace}lang", "eng")

            # Collect English glosses for this sense
            glosses = [g.text.strip() for g in sense.findall("gloss") if g.text]
            meaning = "; ".join(glosses) if glosses else ""

            # Insert into result dict
            if lang_code not in result:
                meta = LANGUAGE_META.get(
                    lang_code,
                    {
                        "language": lang_code,
                        "country": lang_code,
                        "iso2": "XX",
                    },
                )
                result[lang_code] = {
                    "language": meta["language"],
                    "country": meta["country"],
                    "iso2": meta["iso2"],
                    "words": [],
                }

            for kana in katakana_forms:
                key = (kana, lang_code)
                if key not in seen:
                    seen.add(key)
                    result[lang_code]["words"].append(
                        {
                            "katakana": kana,
                            "meaning": meaning,
                        }
                    )

        entry.clear()

    return result


def main():
    parser = argparse.ArgumentParser(description="Export JMdict gairaigo to JSON")
    parser.add_argument(
        "--jmdict",
        default="data/JMdict",
        help="Path to the JMdict XML file (default: data/JMdict)",
    )
    parser.add_argument(
        "--out",
        default="data/gairaigo_full.json",
        help="Output JSON path (default: data/gairaigo_full.json)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=1,
        help="Minimum word count to include a language (default: 1)",
    )
    args = parser.parse_args()

    jmdict_path = Path(args.jmdict)
    if not jmdict_path.exists():
        raise FileNotFoundError(
            f"JMdict file not found at '{jmdict_path}'. "
            "Download it from https://www.edrdg.org/wiki/index.php/JMdict-EDICT_Dictionary_Project "
            "and place it under data/JMdict."
        )

    print(f"Parsing {jmdict_path} ...")
    data = parse_jmdict(jmdict_path)

    # Filter out languages below the threshold
    data = {k: v for k, v in data.items() if len(v["words"]) >= args.min_words}

    # Sort words within each language alphabetically by katakana
    for lang in data.values():
        lang["words"].sort(key=lambda w: w["katakana"])

    # Build a summary for the terminal
    total_words = sum(len(v["words"]) for v in data.values())
    print(f"\n✓ Languages found : {len(data)}")
    print(f"✓ Total entries   : {total_words}")
    print("\nTop 10 by word count:")
    top = sorted(data.items(), key=lambda kv: len(kv[1]["words"]), reverse=True)[:10]
    for code, meta in top:
        print(f"  {code:6s}  {meta['language']:20s}  {len(meta['words']):>5} words")

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved → {out_path}")


if __name__ == "__main__":
    main()
