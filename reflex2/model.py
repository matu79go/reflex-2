"""Reflex-2: fast, local judgments on video, images and audio.

Reflex-2 turns one clip (or image, or audio) into a single 768-d vector with Google EmbeddingGemma 2,
then reads that vector with a small trained head. A head is a logistic regression over the vector:
it trains in seconds from labelled examples and costs almost nothing at inference time.

    rf = Reflex2()                                   # loads google/embeddinggemma-2
    rf.load_heads("heads")                           # e.g. heads/fall.json
    rf.decide("clip.mp4", "fall")                    # {"choice": "fall", "probs": {...}, "latency_ms": ...}
    rf.watch("camera.mp4", "fall")                   # [(t, probs), ...] over a sliding window
    head = rf.train_head({"fall": [...], "adl": [...]})  # your own judgment from examples
"""
from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

PREFIX = "task: classification | query: "
PROMPT = {"video": "A camera video. <|video|>", "image": "A camera image. <|image|>", "audio": "An audio clip. <|audio|>"}
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")


# ---------------------------------------------------------------- inputs

def video_frames(path: str, fps: float = 1.0, max_frames: int = 32, start: float = 0.0, end: float | None = None) -> np.ndarray:
    """Frames sampled evenly over [start, end] at `fps`, at most `max_frames` (the model's default video budget).

    Returns uint8 [N, H, W, 3]. Decoding keeps only the frames it needs."""
    import av

    with av.open(path) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        dur = float(st.duration * st.time_base) if st.duration else float(c.duration / 1e6)
        end = dur if end is None else min(end, dur)
        span = max(end - start, 1e-3)
        n = max(1, min(max_frames, int(span * fps)))
        want = list(start + np.linspace(0, span, n, endpoint=False) + span / n / 2)
        out, last = [], None
        for f in c.decode(video=0):
            last = f
            if f.time is not None and f.time >= want[len(out)]:
                out.append(f.to_ndarray(format="rgb24"))
                if len(out) == n:
                    break
        if not out and last is not None:  # window past the last timestamp: use the last frame
            out.append(last.to_ndarray(format="rgb24"))
    return np.stack(out)


def audio_wave(path: str, sr: int = 16000) -> np.ndarray:
    """Mono float32 at 16 kHz (what the audio encoder expects)."""
    import av

    with av.open(path) as c:
        rs = av.AudioResampler(format="flt", layout="mono", rate=sr)
        chunks = [o.to_ndarray().reshape(-1) for fr in c.decode(audio=0) for o in rs.resample(fr)]
        chunks += [o.to_ndarray().reshape(-1) for o in rs.resample(None)]
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


def _kind(x) -> str:
    if isinstance(x, str):
        e = os.path.splitext(x.lower())[1]
        return "video" if e in VIDEO_EXT else "image" if e in IMAGE_EXT else "audio" if e in AUDIO_EXT else "text"
    if isinstance(x, np.ndarray):
        return "video" if x.ndim == 4 else "image" if x.ndim == 3 else "audio"
    return "image"  # PIL.Image


# ---------------------------------------------------------------- heads

@dataclass
class Head:
    """Logistic-regression head over the embedding. probs = softmax(((x - mu) / sd) @ W + b)."""

    labels: list
    mu: np.ndarray
    sd: np.ndarray
    W: np.ndarray
    b: np.ndarray
    meta: dict = field(default_factory=dict)

    def probs(self, X: np.ndarray) -> np.ndarray:
        z = ((np.atleast_2d(X) - self.mu) / self.sd) @ self.W + self.b
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def save(self, path: str):
        d = {"labels": self.labels, "mu": self.mu.tolist(), "sd": self.sd.tolist(), "W": self.W.tolist(), "b": self.b.tolist(), "meta": self.meta}
        with open(path, "w") as f:
            json.dump(d, f)

    @classmethod
    def load(cls, path: str) -> "Head":
        d = json.load(open(path))
        return cls(d["labels"], np.array(d["mu"], np.float32), np.array(d["sd"], np.float32), np.array(d["W"], np.float32),
                   np.array(d["b"], np.float32), d.get("meta", {}))


def fit_head(X: np.ndarray, y: np.ndarray, labels: list, wd: float = 0.1, steps: int = 300, meta: dict | None = None) -> Head:
    """Multinomial logistic regression with L2, fitted by L-BFGS. Seconds on a CPU for hundreds of examples."""
    X = np.asarray(X, np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-4
    Xt = torch.tensor((X - mu) / sd)
    yt = torch.tensor(np.asarray(y), dtype=torch.long)
    W = torch.zeros(X.shape[1], len(labels), requires_grad=True)
    b = torch.zeros(len(labels), requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt) + wd * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return Head(list(labels), mu, sd, W.detach().numpy(), b.detach().numpy(), meta or {})


# ---------------------------------------------------------------- model

class Reflex2:
    def __init__(self, model: str = "google/embeddinggemma-2", device: str | None = None, modalities=("video", "image", "audio"),
                 fps: float = 1.0, max_frames: int = 32, tokens_per_frame: int | None = None):
        """modalities: drop the ones you do not need to save memory ("audio" -> -300M params, "video"/"image" -> -170M).
        tokens_per_frame: 140 by default (70 to 1120 allowed). On CPU, 70 tokens with max_frames=4 took 7.5 s per clip
        instead of about 11 s, at 91.3% instead of 95.7% balanced accuracy on fall detection."""
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        cfg = {}
        if "audio" not in modalities:
            cfg["audio_config"] = None
        if "video" not in modalities and "image" not in modalities:
            cfg["vision_config"] = None
        self.st = SentenceTransformer(model, device=self.device, model_kwargs={"torch_dtype": dtype}, config_kwargs=cfg or None)
        if tokens_per_frame:
            self.st[0].processor.video_processor.max_soft_tokens = tokens_per_frame
        self.fps, self.max_frames = fps, max_frames
        self.heads: dict[str, Head] = {}

    # -- embeddings
    @torch.no_grad()
    def embed(self, x, kind: str | None = None) -> np.ndarray:
        """One 768-d unit vector for a video (path or [N,H,W,3] frames), an image (path / PIL / [H,W,3]),
        an audio clip (path or 16 kHz float array) or a text."""
        kind = kind or _kind(x)
        if kind == "text":
            inp = PREFIX + x
        else:
            if isinstance(x, str):
                if kind == "video":
                    x = video_frames(x, self.fps, self.max_frames)
                elif kind == "audio":
                    x = audio_wave(x)
                else:
                    from PIL import Image

                    x = Image.open(x).convert("RGB")
            inp = {"text": PREFIX + PROMPT[kind], kind: x}
        return np.asarray(self.st.encode(inp, normalize_embeddings=True), np.float32)

    # -- heads
    def load_heads(self, path: str):
        """A .json head file or a directory of them; the file name (without .json) is the head name."""
        files = sorted(glob.glob(os.path.join(path, "*.json"))) if os.path.isdir(path) else [path]
        for f in files:
            self.heads[os.path.splitext(os.path.basename(f))[0]] = Head.load(f)
        return self

    def _head(self, head) -> Head:
        return self.heads[head] if isinstance(head, str) else head

    def train_head(self, examples: dict, wd: float = 0.1, name: str | None = None) -> Head:
        """examples: {"label": [inputs...], ...}. Inputs are anything embed() accepts. Returns (and registers) a Head."""
        labels = list(examples)
        X = [self.embed(x) for lab in labels for x in examples[lab]]
        y = [i for i, lab in enumerate(labels) for _ in examples[lab]]
        head = fit_head(np.stack(X), np.array(y), labels, wd, meta={"n": len(y), "model": "embeddinggemma-2", "fps": self.fps})
        if name:
            self.heads[name] = head
        return head

    # -- decisions
    def decide(self, x, head, kind: str | None = None) -> dict:
        """Choice + probabilities from a trained head."""
        t0 = time.time()
        r = self.decide_vec(self.embed(x, kind), head)
        r["latency_ms"] = int((time.time() - t0) * 1000)
        return r

    def decide_vec(self, v: np.ndarray, head) -> dict:
        """Same as decide() on an embedding you already have (one embedding can serve many heads)."""
        h = self._head(head)
        probs = {lab: round(float(q), 4) for lab, q in zip(h.labels, h.probs(v)[0])}
        best = max(probs, key=probs.get)
        return {"choice": best, "confidence": probs[best], "probs": probs}

    def zero_shot(self, x, criteria: dict, kind: str | None = None, temperature: float = 0.02) -> dict:
        """Untrained choice by similarity to a short description of each option. Works for search-like questions;
        for reliable judgments, train a head (a few examples per option are often enough)."""
        t0 = time.time()
        r = self.zero_shot_vec(self.embed(x, kind), criteria, temperature)
        r["latency_ms"] = int((time.time() - t0) * 1000)
        return r

    def zero_shot_vec(self, v: np.ndarray, criteria: dict, temperature: float = 0.02) -> dict:
        names = list(criteria)
        T = np.stack([self.embed(criteria[n] or n, "text") for n in names])
        z = (T @ v) / temperature
        p = np.exp(z - z.max())
        p /= p.sum()
        probs = {n: round(float(q), 4) for n, q in zip(names, p)}
        best = max(probs, key=probs.get)
        return {"choice": best, "confidence": probs[best], "probs": probs}

    def watch(self, path: str, head, window: float = 4.0, step: float = 0.5, fps: float = 2.0) -> list:
        """Score a video the way a live camera would: every `step` seconds, judge the last `window` seconds.
        Returns [(t, probs), ...]."""
        import av

        with av.open(path) as c:
            st = c.streams.video[0]
            dur = float(st.duration * st.time_base) if st.duration else float(c.duration / 1e6)
        h, out, t = self._head(head), [], min(1.0, dur)
        while t <= dur + 1e-6:
            v = video_frames(path, fps, int(window * fps), max(0.0, t - window), t)
            p = h.probs(self.embed(v, "video"))[0]
            out.append((round(t, 2), {lab: round(float(q), 4) for lab, q in zip(h.labels, p)}))
            t += step
        return out
