---
license: apache-2.0
base_model: google/embeddinggemma-2
library_name: sentence-transformers
pipeline_tag: video-classification
tags:
  - reflex-2
  - fall-detection
  - video-classification
  - embeddinggemma
  - edge-ai
  - on-device
language:
  - en
---

# Reflex-2

**Fast, local judgments on video. 740M parameters, one GPU, no cloud.**

Reflex-2 judges camera video, images and audio on your own hardware. It turns a clip into one vector with [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) and reads that vector with a small head trained from examples. This repository holds the heads; the base model is loaded from Google's repository and is not modified.

- **Fall detection** (`fall.json`): 95.7% balanced accuracy on people the head never saw, 0.3 s per clip on one GPU.
- **Faster than a frontier LLM.** On the same 40 fall clips: Reflex-2 97.5% in 0.28 s per clip, Gemini 3.8 Flash 85.0% in 6.95 s.
- **Your own judgment in seconds** from a few dozen labelled clips.

Code, demos and training scripts: [github.com/matu79go/reflex-2](https://github.com/matu79go/reflex-2) · Sibling model for text and images: [Reflex-1](https://huggingface.co/matu79go/Reflex-1-4B)

## Quickstart

```bash
pip install "sentence-transformers>=6.1" "transformers>=5.19" av
git clone https://github.com/matu79go/reflex-2 && cd reflex-2
```

```python
from huggingface_hub import hf_hub_download
from reflex2 import Reflex2, Head

rf = Reflex2()                                                    # google/embeddinggemma-2
fall = Head.load(hf_hub_download("matu79go/Reflex-2", "fall.json"))
rf.decide("clip.mp4", fall)
# {'choice': 'fall', 'confidence': 0.97, 'probs': {'adl': 0.03, 'fall': 0.97}, 'latency_ms': 290}
rf.watch("camera.mp4", fall)   # judge the last 4 s every 0.5 s, like a live camera
```

## Heads

| File | Judgment | Labels | Training data | Accuracy (leave-one-person-out) |
|---|---|---|---|---|
| `fall.json` | a person falls, indoor home camera | `fall`, `adl` (daily activity) | GMDCSA24, 160 clips, 4 actors, MIT License | 95.7% balanced, AUC 0.990 |

A head is a logistic regression over the 768-d embedding of a clip (frames at 1 per second, up to 32), stored as JSON (labels, normalization, weights).

## Evaluation

**Fall detection, GMDCSA24** (4 actors in 3 homes; each person held out in turn)

| Examples per class | Balanced accuracy | AUC |
|---|---|---|
| none (similarity to a text description) | n/a | 0.836 |
| 5 | 85.6% | 0.939 |
| 20 | 93.3% | 0.984 |
| all | **95.7%** | **0.990** |

**Against a frontier LLM, same clips and frames**

| | Reflex-2 | Gemini 3.8 Flash |
|---|---|---|
| Fall detection, 40 clips: accuracy | **97.5%** | 85.0% |
| Fall detection: median time per clip | **0.28 s** | 6.95 s |
| Surveillance anomalies (UCF-Crime), 40 clips: accuracy | 95% | 95% |
| Surveillance anomalies: median time per clip | **1.4 s** | 16.1 s |

Reflex-2 on one NVIDIA GB10 (ASUS Ascent GX10), one clip at a time; Gemini via OpenRouter, network included, thinking minimal. The surveillance head is not released (UCF-Crime is for research use).

**Speed and memory**: 0.30 s per 7 s clip and 2.2 GB of GPU memory on one GB10. CPU only (4 Arm Cortex-X925 cores): about 11 s per clip, 7.5 s with 70 tokens per frame and 4 frames.

## Limits

- Detects **when** something happens, not **who**; no bounding boxes.
- Heads are trained per judgment; without a head only similarity search is available.
- Does not reason over written instructions (use Reflex-1 for that).
- Evaluated on public, staged or YouTube footage. Check accuracy on your own cameras before relying on it.

## License

Apache 2.0. Base model: Google EmbeddingGemma 2 (Apache 2.0); Reflex-2 is not affiliated with or endorsed by Google. `fall.json` is trained on GMDCSA24 (MIT License).
