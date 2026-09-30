"""Ready-made clips: Pollen's emotion and dance libraries, downloaded from the Hugging Face Hub on first use.

``clips.cheerful1`` or ``clips.get("cheerful1")`` gives a ``Clip``; ``clips.names()`` lists them all.
"""

from __future__ import annotations

import functools
import logging
import os
import tempfile
import urllib.request
from pathlib import Path

from reachy_animation.motion import Clip

logger = logging.getLogger(__name__)

# Pinned revisions, so a clip never changes under a caller. Names are the Hub file stems.
_LIBRARIES = {
    ("pollen-robotics/reachy-mini-emotions-library", "873ae49f0b89114b7e535eff0c1f7560d21d9357"): (
        "amazed1 anxiety1 attentive1 attentive2 boredom1 boredom2 calming1 cheerful1 come1 confused1 contempt1 "
        "curious1 dance1 dance2 dance3 disgusted1 displeased1 displeased2 downcast1 dying1 electric1 enthusiastic1 "
        "enthusiastic2 exhausted1 fear1 frustrated1 furious1 go_away1 grateful1 helpful1 helpful2 impatient1 "
        "impatient2 incomprehensible2 indifferent1 inquiring1 inquiring2 inquiring3 irritated1 irritated2 laughing1 "
        "laughing2 lonely1 lost1 loving1 mini-deep-sleep no1 no_excited1 no_sad1 oops1 oops2 proud1 proud2 proud3 "
        "rage1 relief1 relief2 reprimand1 reprimand2 reprimand3 resigned1 sad1 sad2 scared1 serenity1 shy1 sleep1 "
        "success1 success2 surprised1 surprised2 thoughtful1 thoughtful2 tired1 toc-toc-toc uncertain1 "
        "uncomfortable1 understanding1 understanding2 waiting wake-mini-up welcoming1 welcoming2 yes1 yes_sad1"
    ),
    ("pollen-robotics/reachy-mini-dances-library", "3564295e72d41c1271f46bf5540a3fcad9d5f669"): (
        "chicken_peck chin_lead dizzy_spin grid_snap groovy_sway_and_roll head_tilt_roll interwoven_spirals "
        "jackson_square neck_recoil pendulum_swing polyrhythm_combo sharp_side_tilt side_glance_flick side_peekaboo "
        "side_to_side_sway simple_nod stumble_and_recover uh_huh_tilt yeah_nod"
    ),
}
# Python names (hyphens become underscores) -> (repo, revision, file stem).
_SOURCES = {
    stem.replace("-", "_"): (repo, revision, stem)
    for (repo, revision), stems in _LIBRARIES.items()
    for stem in stems.split()
}
_CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "reachy-animation" / "clips"


def names() -> list[str]:
    """Every available clip name, sorted."""
    return sorted(_SOURCES)


@functools.cache
def get(name: str) -> Clip:
    """The clip called ``name``, downloaded and cached on first use (so play it with ``play(lambda: ...)``)."""
    if name not in _SOURCES:
        raise KeyError(f"no clip named {name!r}; see clips.names()")
    repo, revision, stem = _SOURCES[name]
    path = _CACHE / revision / f"{stem}.json"
    if not path.exists():
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{stem}.json"
        logger.info("downloading clip %s from %s", name, url)
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        # Write then rename, so a concurrent or interrupted download never leaves a partial file behind.
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            tmp.write(data)
        os.replace(tmp.name, path)
    clip = Clip.load(path)
    clip.name = name
    return clip


def __getattr__(name: str) -> Clip:
    if name not in _SOURCES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return get(name)


def __dir__() -> list[str]:
    return [*globals(), *_SOURCES]
