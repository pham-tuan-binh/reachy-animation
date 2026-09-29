"""Validate animation offline: script a scenario, tick the animator on simulated time, render it in MuJoCo.

  python -m reachy_animation.sim --out demo.mp4 --speech speech.wav@1.0 --play fear1.json@2.5 --queue x.json@3

Playback mirrors the SDK's MuJoCo backend: the SDK's analytical IK drives the Stewart platform, antennas are
negated, and an unreachable pose holds the last reachable command like the daemon. Speech is fed in 100 ms
chunks all at once, as a realtime API bursts audio ahead of playback. Needs the ``sim`` extra.
"""

from __future__ import annotations

import argparse
import json
import logging
import wave
from collections.abc import Callable
from functools import partial
from importlib.resources import files
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
import reachy_mini
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from reachy_mini_rust_kinematics import ReachyMiniRustKinematics

from reachy_animation.animator import Animator
from reachy_animation.motion import Clip
from reachy_animation.pose import Pose, head_matrix, zero_pose

logger = logging.getLogger(__name__)

PHYSICS_SUBSTEPS = 8


class Sim:
    """Drive the MuJoCo model with a pose per frame and render each frame."""

    def __init__(self, fps: float, width: int = 640, height: int = 480) -> None:
        """Load the SDK's model and kinematics; physics steps ``PHYSICS_SUBSTEPS`` times per frame."""
        root = Path(str(files(reachy_mini)))
        self.model = mujoco.MjModel.from_xml_path(str(root / "descriptions/reachy_mini/mjcf/scenes/empty.xml"))
        self.model.opt.timestep = 1.0 / (fps * PHYSICS_SUBSTEPS)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.distance, self.camera.elevation, self.camera.azimuth = 0.52, -6.0, 158.0
        self.camera.lookat[:] = [0.0, 0.0, 0.16]

        kinematics = json.loads((root / "assets/kinematics_data.json").read_text())
        self._head_z_offset = kinematics["head_z_offset"]
        self._kinematics = ReachyMiniRustKinematics(kinematics["motor_arm_length"], kinematics["rod_length"])
        for motor in kinematics["motors"]:
            self._kinematics.add_branch(
                motor["branch_position"],
                np.linalg.inv(motor["T_motor_world"]).tolist(),
                1 if motor["solution"] else -1,
            )
        self.unreachable = 0

    def joints(self, pose: Pose) -> NDArray[np.float64]:
        """Seven joint targets ``[body_yaw, stewart_1..6]`` for ``pose``; NaN when unreachable."""
        head = head_matrix(pose)
        head[2, 3] += self._head_z_offset
        return np.array(
            self._kinematics.inverse_kinematics_safe(
                head.tolist(), body_yaw=float(pose[8]), max_relative_yaw=np.deg2rad(65), max_body_yaw=np.deg2rad(160)
            )
        )

    def render(self, poses: NDArray[np.float64]) -> list[NDArray[np.uint8]]:
        """Play ``poses`` (T, 9) one per frame and return the rendered RGB frames."""
        neutral = self.joints(zero_pose())
        last = self.joints(poses[0])
        if not np.all(np.isfinite(last)):
            last = neutral
        self._reset(np.concatenate([last, -poses[0, 6:8]]), np.concatenate([neutral, [0.0, 0.0]]))
        self.unreachable = 0
        frames = []
        for pose in poses:
            q = self.joints(pose)
            if np.all(np.isfinite(q)):
                last = q
            else:
                self.unreachable += 1
            self.data.ctrl[:] = np.concatenate([last, -pose[6:8]])
            for _ in range(PHYSICS_SUBSTEPS):
                mujoco.mj_step(self.model, self.data)
            self.renderer.update_scene(self.data, self.camera)
            frames.append(self.renderer.render())
        if self.unreachable:
            logger.warning("%d/%d frames unreachable, held the last reachable command", self.unreachable, len(poses))
        return frames

    def _reset(
        self, target: NDArray[np.float64], neutral: NDArray[np.float64], ramp: int = 300, settle: int = 100
    ) -> None:
        """Settle at neutral with contacts off, then ramp: snapping to a target can close the linkage mirrored."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        contacts = (m.geom_contype.copy(), m.geom_conaffinity.copy())
        m.geom_contype[:] = 0
        m.geom_conaffinity[:] = 0
        d.ctrl[:] = neutral
        mujoco.mj_forward(m, d)
        for _ in range(settle):
            mujoco.mj_step(m, d)
        for k in range(1, ramp + 1):
            d.ctrl[:] = neutral + (target - neutral) * (k / ramp)
            mujoco.mj_step(m, d)
        for _ in range(settle):
            mujoco.mj_step(m, d)
        m.geom_contype[:], m.geom_conaffinity[:] = contacts


def read_wav(path: str | Path) -> tuple[NDArray[np.int16], int]:
    """Read a 16-bit PCM WAV file as mono int16 samples and its sample rate."""
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {8 * w.getsampwidth()}-bit")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
        return pcm.mean(axis=1).astype(np.int16), w.getframerate()


def write_video(
    frames: list[NDArray[np.uint8]],
    path: str | Path,
    fps: float,
    audio: NDArray[np.int16] | None = None,
    sample_rate: int = 0,
) -> None:
    """Encode ``frames`` to mp4, muxing a mono int16 ``audio`` track when given."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = path.with_suffix(".wav")
    if audio is not None:
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(audio.tobytes())
    height, width = frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        quality=8,
        macro_block_size=1,
        audio_path=str(wav) if audio is not None else None,
        audio_codec="aac" if audio is not None else None,
    )
    writer.send(None)
    for frame in frames:
        writer.send(np.ascontiguousarray(frame))
    writer.close()
    wav.unlink(missing_ok=True)


def _at(spec: str) -> tuple[str, float]:
    path, _, at = spec.rpartition("@")
    if not path:
        raise argparse.ArgumentTypeError(f"expected PATH@SECONDS, got {spec!r}")
    return path, float(at)


def _caption(frame: NDArray[np.uint8], text: str) -> NDArray[np.uint8]:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, image.width, 20], fill=(0, 0, 0))
    draw.text((6, 4), text, fill=(255, 230, 0))
    return np.asarray(image)


def main() -> None:
    """Run the scripted scenario and write the video."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--speech", type=_at, metavar="WAV@T", help="16-bit WAV spoken from T")
    parser.add_argument("--interrupt-speech", type=float, metavar="T", help="cut the speech at T")
    parser.add_argument("--play", type=_at, action="append", default=[], metavar="MOVE_JSON@T")
    parser.add_argument("--queue", type=_at, action="append", default=[], metavar="MOVE_JSON@T")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    now = 0.0
    animator = Animator(fps=args.fps, clock=lambda: now)
    events: list[tuple[float, str, Callable[[], None]]] = []
    audio, sample_rate = None, 0
    if args.speech:
        path, at = args.speech
        pcm, sample_rate = read_wav(path)
        chunk = sample_rate // 10
        for i in range(0, len(pcm), chunk):
            events.append(
                (at, "speech" if i == 0 else "", partial(animator.feed_speech, pcm[i : i + chunk], sample_rate))
            )
        end = len(pcm) if args.interrupt_speech is None else max(0, int((args.interrupt_speech - at) * sample_rate))
        audio = np.zeros(int(args.duration * sample_rate), dtype=np.int16)
        start = int(at * sample_rate)
        track = pcm[:end][: max(0, len(audio) - start)]
        audio[start : start + len(track)] = track
    if args.interrupt_speech is not None:
        events.append((args.interrupt_speech, "interrupt speech", animator.interrupt_speech))
    for path, at in args.play:
        events.append((at, f"play {Path(path).stem}", partial(animator.play, Clip.load(path))))
    for path, at in args.queue:
        events.append((at, f"queue {Path(path).stem}", partial(animator.play, Clip.load(path), queue=True)))
    events.sort(key=lambda e: e[0])

    poses, captions = [], []
    for k in range(round(args.duration * args.fps)):
        now = k / args.fps
        while events and events[0][0] <= now:
            _, label, fire = events.pop(0)
            fire()
            if label:
                logger.info("%5.2fs %s", now, label)
        poses.append(animator.tick(now))
        playing = animator.playing
        captions.append(f"{now:5.2f}s  {playing.name if playing else 'idle'}")

    frames = Sim(args.fps).render(np.array(poses))
    write_video([_caption(f, c) for f, c in zip(frames, captions)], args.out, args.fps, audio, sample_rate)
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
