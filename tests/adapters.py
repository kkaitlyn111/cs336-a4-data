from __future__ import annotations

import os
from typing import Any
from cs336_data.text_extractor import extract_text
from cs336_data.language_identifier import find_language_scores
from cs336_data.personal_mask import mask_emails, mask_phone_numbers, mask_ips
from cs336_data.nsfw_toxic import find_nsfw, find_toxic
from cs336_data.quality_filter import passes_quality_filters
from cs336_data.dedup import exact_deduplication
from cs336_data.minhashLSHdedup import run_minhash_deduplication as minhash_impl

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return find_language_scores(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return find_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return find_toxic(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    return NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    return passes_quality_filters(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    return exact_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    return minhash_impl(
        filepaths=input_files,
        n=ngrams,
        k=num_hashes,
        bands=num_bands,
        threshold=jaccard_threshold,
        output_dir=output_directory,
    )
