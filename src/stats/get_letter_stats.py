"""Letter-level script statistics utilities.

Includes functions for:
- Stable text normalization,
- Unicode-based script detection (Cyrillic, Latin, Chinese),
- Calculation of ratios over alphabetic characters only.
"""

CYRILLIC_START = 0x0400
CYRILLIC_END = 0x04FF

LATIN_UPPER_START = 0x0041
LATIN_UPPER_END = 0x005A

LATIN_LOWER_START = 0x0061
LATIN_LOWER_END = 0x007A

CJK_UNIFIED_START = 0x4E00
CJK_UNIFIED_END = 0x9FFF

CJK_EXTENSION_A_START = 0x3400
CJK_EXTENSION_A_END = 0x4DB5


def normalize_text(s: str | None) -> str:
    """Minimal normalization for stable matching.

    - Convert None -> "";
    - Strip trailing/leading whitespace.
    """
    s2 = "" if s is None else str(s)
    return s2.strip()


def is_cyrillic(ch: str) -> bool:
    """Check whether a character belongs to the Cyrillic Unicode block."""
    code = ord(ch)
    return CYRILLIC_START <= code <= CYRILLIC_END


def is_latin(ch: str) -> bool:
    """Check whether a character belongs to the Latin Unicode block."""
    code = ord(ch)
    return LATIN_UPPER_START <= code <= LATIN_UPPER_END or LATIN_LOWER_START <= code <= LATIN_LOWER_END


def is_chinese(ch: str) -> bool:
    """Check whether a character belongs to the Chinese Unicode block."""
    code = ord(ch)
    return CJK_UNIFIED_START <= code <= CJK_UNIFIED_END or CJK_EXTENSION_A_START <= code <= CJK_EXTENSION_A_END


def calc_letter_level_ratios(
    text: str | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Compute ratios over letters only.

    Returns (cyr, lat, chi, other) or (None,...,None) if no letters/empty.
    """
    if text is None:
        return None, None, None, None

    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return None, None, None, None

    cyr = lat = chi = other = 0
    for ch in letters:
        if is_cyrillic(ch):
            cyr += 1
        elif is_latin(ch):
            lat += 1
        elif is_chinese(ch):
            chi += 1
        else:
            other += 1

    total = len(letters)
    return cyr / total, lat / total, chi / total, other / total
