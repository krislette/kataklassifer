"""
Parses the JMdict XML file and extracts gairaigo (loanword) entries.

A word is considered gairaigo if at least one of its <sense> elements
contains an <lsource> tag. This is JMdict's convention for marking
foreign-origin words. The <lsource> tag also carries the origin language
via its xml:lang attribute (defaults to English when the attribute is absent).

Each extracted record is a (katakana, language) pair, where:
  - katakana : the loanword's pure-katakana written form
  - language : the ISO 639-2 language code of the donor language
"""

import re
import pandas as pd
from lxml import etree

# Matches strings made entirely of katakana characters (U+30A0–U+30FF).
# The prolonged sound mark ー (U+30FC) is included in that range.
KATAKANA_PATTERN = re.compile(r"^[\u30A0-\u30FF]+$")

# The xml:lang attribute is stored under the full XML namespace URI by lxml.
# The xml namespace (http://www.w3.org/XML/1998/namespace) is always defined
# in XML and requires no declaration, so lxml resolves it even with DTD
# entity resolution disabled.
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def load_gairaigo(filepath: str) -> pd.DataFrame:
    """
    Parse JMdict and return a DataFrame of gairaigo entries.

    JMdict uses DOCTYPE entity references that cause standard XML parsers to
    crash. Setting resolve_entities=False and recover=True lets lxml skip
    those definitions while still parsing the rest of the document correctly.

    Filtering rules applied here:
      - Only entries with at least one <lsource> tag are kept (loanwords only).
      - Only entries with a pure-katakana written form are kept.

    Args:
        filepath : Path to the JMdict file (with or without .xml extension).

    Returns:
        DataFrame with columns ['katakana', 'language'].
    """
    parser = etree.XMLParser(resolve_entities=False, recover=True)
    tree = etree.parse(filepath, parser)
    root = tree.getroot()

    records = []

    for entry in root.findall("entry"):
        # Try to determine the donor language, none means not a loanword
        origin_lang = _get_origin_language(entry)
        if origin_lang is None:
            continue

        # Try to get a pure-katakana form, none means no usable representation
        katakana = _get_katakana_form(entry)
        if katakana is None:
            continue

        records.append({"katakana": katakana, "language": origin_lang})

    return pd.DataFrame(records)


def _get_origin_language(entry) -> str | None:
    """
    Extract the donor language code from the first <lsource> found in any sense.

    JMdict omits the xml:lang attribute when the source language is English,
    so we default to 'eng' in that case. Returns None if no <lsource> exists,
    which means the entry is a native Japanese word, not a loanword.
    """
    for sense in entry.findall("sense"):
        lsource = sense.find("lsource")
        if lsource is not None:
            return lsource.get(XML_LANG, "eng")
    return None


def _get_katakana_form(entry) -> str | None:
    """
    Return the first pure-katakana written form of a JMdict entry.

    JMdict organizes written forms in two layers:
      - <k_ele> / <keb>: kanji element body (the "dictionary" written form).
      - <r_ele> / <reb>: reading element body (always kana).

    Gairaigo are usually written in katakana, so we check <keb> first,
    then fall back to <reb> for entries that only have kana readings.
    """
    # Check kanji element bodies first (keb)
    for k_ele in entry.findall("k_ele"):
        keb = k_ele.findtext("keb", default="")
        if KATAKANA_PATTERN.match(keb):
            return keb

    # Fall back to reading element bodies (reb)
    for r_ele in entry.findall("r_ele"):
        reb = r_ele.findtext("reb", default="")
        if KATAKANA_PATTERN.match(reb):
            return reb

    return None
