"""Split text into overlapping chunks along natural boundaries.

Strategy (a simplified "recursive" splitter):
  1. Split into paragraphs, then sentences; sentences longer than `chunk_size`
     are hard-split by characters.
  2. Greedily pack these units into chunks of at most `chunk_size` characters.
  3. Start each new chunk with the trailing units of the previous one (up to
     `chunk_overlap` characters) so a fact sitting on a boundary appears whole
     in at least one chunk.
"""

import re


def split_into_units(text: str, chunk_size: int) -> list[str]:
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())  # normalise whitespace / PDF line breaks
        if not paragraph:
            continue
        # Sentences are the packing unit: small enough to make overlap meaningful.
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            if len(sentence) <= chunk_size:
                units.append(sentence)
            else:
                units.extend(sentence[i : i + chunk_size] for i in range(0, len(sentence), chunk_size))
    return units


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for unit in split_into_units(text, chunk_size):
        if current and length + len(unit) + 1 > chunk_size:
            chunks.append(" ".join(current))
            # Carry over the tail of the previous chunk as overlap.
            tail: list[str] = []
            tail_len = 0
            for previous in reversed(current):
                if tail_len + len(previous) + 1 > chunk_overlap:
                    break
                tail.insert(0, previous)
                tail_len += len(previous) + 1
            # Overlap is a bonus: drop it rather than push the next chunk past chunk_size.
            while tail and tail_len + len(unit) + 1 > chunk_size:
                tail_len -= len(tail.pop(0)) + 1
            current, length = tail, tail_len
        current.append(unit)
        length += len(unit) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks
