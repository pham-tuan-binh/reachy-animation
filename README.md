# reachy-animation

Makes [Reachy Mini](https://github.com/pollen-robotics/reachy_mini) move naturally while it talks and emotes.
You give it **the audio being spoken** and **the motions to play**. It gives back one full-body pose per
frame, 60 times a second by default.

```bash
pip install git+https://github.com/pham-tuan-binh/reachy-animation
```

```python
from reachy_animation import Animator, Clip, to_target

animator = Animator(fps=60)
animator.start(lambda pose: robot.set_target(*to_target(pose)))

animator.feed_speech(chunk, 24_000)          # each audio chunk you send to the speaker: the head sways with it
animator.play(Clip.load("cheerful1.json"))   # play a motion now
```

That's the whole idea. The rest is reference.

## Example

Everything in one place: a custom motion, clips at different rates, queueing, speech and barge-in.

```python
import math
import time

import numpy as np
from reachy_mini import ReachyMini
from reachy_animation import Animator, Breathing, Clip, to_target


class Nod:
    """A custom motion: anything with name, duration and sample(t) -> pose."""

    name, duration = "nod", 1.0

    def sample(self, t):
        pose = np.zeros(9)                                     # x y z roll pitch yaw antenna_r antenna_l body_yaw
        pose[4] = 0.25 * math.sin(math.pi * min(t, 1.0))       # pitch down and back up
        return pose


robot = ReachyMini()
animator = Animator(fps=60, idle=Breathing(), blend_s=0.4)
animator.start(lambda pose: robot.set_target(*to_target(pose)))    # 60 poses per second to the robot

animator.play(Clip.load("cheerful1.json"))                     # a 50 fps library clip, resampled to 60
animator.play(Clip.from_frames(np.zeros((50, 9)), fps=25), queue=True)   # 2 s of a 25 fps clip, after it
animator.play(Nod(), queue=True)                               # then the custom nod

speech = (3000 * np.sin(np.arange(3 * 24_000) / 8)).astype(np.int16)   # stand-in for TTS audio
for chunk in np.array_split(speech, 30):                       # chunks may arrive faster than they play
    animator.feed_speech(chunk, 24_000)                        # the head sways with it, on top of the clips

time.sleep(2)
animator.interrupt_speech()                                    # the user barged in: stop swaying
animator.stop()                                                # drop the queue, blend back to idle
time.sleep(1)
animator.close()
```

## Speech

| Call | When |
|---|---|
| `feed_speech(pcm, sample_rate)` | every chunk of speech audio, as you queue it for playback |
| `interrupt_speech()` | playback was cut (the user barged in) |

Chunks can arrive faster than they play (realtime APIs burst). They're lined up back to back, so the sway
follows what is *heard*.

## Motion

| Call | Effect |
|---|---|
| `play(motion)` | play now, replacing whatever is playing or queued |
| `play(motion, queue=True)` | play after what is already queued |
| `stop()` | back to idle |

Every change crossfades, so the robot never jumps.

```python
Clip.load("fear1.json")               # a recorded-move file (Pollen's emotion/dance libraries)
Clip.load(api_response["moves"][0])   # the same format as a dict (the motion-generator API)
Clip.from_frames(array, fps=25)       # (N, 9) poses, with the fps they were recorded at
Clip.from_sdk(sdk_move)               # any SDK Move, e.g. a DanceMove
```

**One rate.** `fps` is both the control rate and the animation frame rate. Clips can be recorded at any rate.
`play` resamples them to `fps` (smoothing first when downsampling), so timing is kept and every tick lands on a
frame.

Need something else, like procedural motion or a stream from a generator? Any object with `name`,
`duration` (use `math.inf` if it ends only when interrupted) and `sample(t) -> pose` can be played.

## Poses

A pose is a numpy 9-vector, the same layout as the motion generator's trajectories:

```
x  y  z  (m)  |  roll  pitch  yaw  (rad)  |  antenna_right  antenna_left  (rad)  |  body_yaw  (rad)
```

`to_target(pose)` gives `(head_4x4, antennas, body_yaw)` for `set_target`; `from_target` goes the other way.

## Options

```python
Animator(
    fps=60,                # poses per second
    idle=Breathing(),      # what plays when nothing else does
    blend_s=0.4,           # crossfade duration
    speech_latency_s=0.0,  # delay from feed_speech to sound coming out of the speaker
)
```

`animator.tick(t)` returns the pose at time `t` if you'd rather drive the loop yourself; `close()` stops
`start()`'s thread.

## Try it in simulation

```bash
pip install "reachy-animation[sim] @ git+https://github.com/pham-tuan-binh/reachy-animation"
python -m reachy_animation.sim --out demo.mp4 --duration 10 \
    --speech speech.wav@1.0 --play cheerful1.json@2.5 --queue startled__0.json@3.0
```

This renders the official MuJoCo model with the SDK's IK, with the speech audio muxed in and a caption showing
what's playing. On macOS, `say -o speech.wav --data-format=LEI16@24000 "Hi!"` makes a test WAV.

**Not handled here: reachability.** Some library clips (e.g. `cheerful1`) ask for poses the robot can't
reach. The sim, like the daemon, holds the last reachable pose. A real robot link should project poses first
(e.g. the motion generator's `common/reach.py`).

## Develop

```bash
uv sync --extra sim
uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy src
```
