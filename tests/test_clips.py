import json
from pathlib import Path

import numpy as np
import pytest

from reachy_animation import Clip, clips, to_target


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(clips, "_CACHE", tmp_path)
    clips.get.cache_clear()
    return tmp_path


def _cache_move(cache: Path, name: str) -> None:
    _, revision, stem = clips._SOURCES[name]
    frame = {"head": to_target(np.zeros(9))[0].tolist(), "antennas": [0.0, 0.0], "body_yaw": 0.0}
    move = {"description": "test", "time": [0.0, 1.0], "set_target_data": [frame, frame]}
    (cache / revision).mkdir(exist_ok=True)
    (cache / revision / f"{stem}.json").write_text(json.dumps(move))


def test_names_cover_both_libraries_as_python_names() -> None:
    names = clips.names()
    assert {"cheerful1", "fear1", "simple_nod", "toc_toc_toc"} <= set(names)
    assert names == sorted(names)
    assert all(name.isidentifier() for name in names)
    assert set(names) <= set(dir(clips))


def test_clips_load_from_the_cache_by_name_or_attribute(cache: Path) -> None:
    _cache_move(cache, "cheerful1")
    _cache_move(cache, "toc_toc_toc")
    clip = clips.cheerful1
    assert isinstance(clip, Clip)
    assert clip.name == "cheerful1"
    assert clip.duration == 1.0
    assert clips.get("cheerful1") is clip
    assert clips.toc_toc_toc.name == "toc_toc_toc"


def test_unknown_clips_raise() -> None:
    with pytest.raises(KeyError, match="clips.names"):
        clips.get("nope")
    with pytest.raises(AttributeError):
        clips.nope  # noqa: B018
