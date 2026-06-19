"""Utility functions for dataset creation scripts."""

import html
import re
from pathlib import Path


def clean_text(text: str) -> str:
    """Clean text.

    - Remove HTML tags like <br>, <li>, etc.
    - Replace HTML entities (&nbsp;, &lt;, etc.).
    - Remove pseudo-tags like ;li; ;/li;.
    - Collapse multiple spaces and newlines.
    """
    if not isinstance(text, str):
        return text
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r";/?[a-zA-Z0-9]+;", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def clean_srt_text(srt_text: str) -> str:
    """Remove indices and timestamps from SRT content."""
    lines = []
    for raw_line in srt_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # timestamp
        if re.match(r"\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}", line):
            continue
        # numeric index
        if line.isdigit():
            continue
        lines.append(line)
    return " ".join(lines)


def get_files(input_dir: Path) -> list[Path]:
    """Collect all supported data files (Parquet and JSONL) from a directory."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError("input_dir must be an existing directory")

    supported_ext = ("*.parquet", "*.jsonl")
    files: list[Path] = []

    for ext in supported_ext:
        files.extend(input_dir.glob(ext))

    files = sorted(files)
    if not files:
        raise ValueError(f"No supported files found in {input_dir}")

    return files
