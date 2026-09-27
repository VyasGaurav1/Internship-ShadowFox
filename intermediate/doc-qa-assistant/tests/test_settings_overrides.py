"""Regression tests for the Settings override path (sidebar sliders -> chunking).

These pin the exact defect reported from a real run:

    TypeError: Settings.__init__() got an unexpected keyword argument 'overlap'
    app.py:77  local_settings = replace(settings, chunk_size=..., overlap=...)

The dataclass field is ``chunk_overlap``; every override now goes through
``config.with_overrides`` and is covered here.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from config import load_settings, with_overrides
from ingest import Document
from qa import build_index

DOC = (
    "Zorblax is a fictional calibration unit used only in the Alpha-9 experiment. "
    "A Zorblax reading above 40 indicates the Orbital Loom needs recalibration. "
    "Calibration is performed by the Qarn tool every 500 cycles. "
) * 4


def test_overrides_use_real_field_names():
    settings = load_settings()
    local = with_overrides(settings, chunk_size=120, chunk_overlap=10, top_k=2)
    assert local.chunk_size == 120
    assert local.chunk_overlap == 10
    assert local.top_k == 2


def test_reported_typeerror_signature_is_gone():
    """`overlap=` was the bug; `chunk_overlap=` is the fix."""
    settings = load_settings()

    with pytest.raises(TypeError):
        # audit-expected: this is the intentional negative test that pins the bug
        replace(settings, chunk_size=120, overlap=10)  # what the app used to do

    local = with_overrides(settings, chunk_size=120, chunk_overlap=10)  # what it does now
    assert local.chunk_overlap == 10


def test_overrides_reach_the_index_statistics():
    """The slider values must actually reach the chunking logic."""
    settings = load_settings()
    local = with_overrides(settings, chunk_size=100, chunk_overlap=20)
    index = build_index([Document("d.txt", "d.txt", DOC, "text")], local)
    assert index.stats["chunk_size"] == 100
    assert index.stats["chunk_overlap"] == 20


def test_overrides_only_touch_provided_fields():
    settings = load_settings()
    local = with_overrides(settings, top_k=3)
    assert local.top_k == 3
    assert local.chunk_size == settings.chunk_size
    assert local.chunk_overlap == settings.chunk_overlap


def test_no_changes_returns_the_same_object():
    settings = load_settings()
    assert with_overrides(settings) is settings


def test_all_settings_fields_are_addressable_by_with_overrides():
    """Every field the UI can override must exist on the dataclass."""
    from dataclasses import fields

    names = {field.name for field in fields(load_settings())}
    for candidate in ("chunk_size", "chunk_overlap", "top_k"):
        assert candidate in names
