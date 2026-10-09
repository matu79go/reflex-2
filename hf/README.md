---
license: apache-2.0
base_model: google/embeddinggemma-2
library_name: sentence-transformers
pipeline_tag: video-classification
tags:
  - reflex-2
  - embeddinggemma
  - fall-detection
  - video-classification
  - audio-classification
  - edge-ai
  - on-device
language:
  - en
---

# Reflex-2

**Google EmbeddingGemma 2, improved into a video judgment model that runs on phones. Judges video at Jev speed with frontier-LLM-level accuracy or better.**

[EmbeddingGemma 2](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/) (Google, Apache 2.0, 740M parameters) turns text, images, video and audio into one vector. It is a search model: as is, it cannot judge whether a clip shows a fall. Reflex-2 turns it into a judgment model by training a small head from examples, so it can tell, on the device, the moment something happens.

- **Falls**: caught 1.2 to 2.1 s after the person falls, while Gemini 3.8 Flash takes 8.3 s or misses it. 0.28 s per judgment on one GPU.
- **Surveillance anomalies**: the same accuracy as Gemini 3.8 Flash (95%), 11× faster.
- **Sounds** (crying baby, glass breaking, siren, alarm...): 100% vs 85% for Gemini on the same 40 clips, 0.05 s vs 3.66 s.
- **Video stays on the device.** 1.5 GB of weights, no per-call fee.

This repository holds the trained heads; the base model is loaded from [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) and is not modified.

Code, demos and training scripts: [github.com/matu79go/reflex-2](https://github.com/matu79go/reflex-2) · Article: [suzuki-shoten.dev/projects/reflex-2](https://suzuki-shoten.dev/projects/reflex-2/) · Sibling model for text and images: [Reflex-1](https://huggingface.co/matu79go/Reflex-1-4B)

## Demo

How fast does it notice that someone fell? The clock starts at the moment of the fall; both models judge the same frames (the last 4 s).

![Fall detection race](https://raw.githubusercontent.com/matu79go/reflex-2/main/media/race2_en.gif)

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

# your own judgment from examples (video, image or audio)
head = rf.train_head({"glass": ["g1.wav", "g2.wav"], "other": ["o1.wav", "o2.wav"]})
```

Or try it in the browser: [Open in Colab](https://colab.research.google.com/github/matu79go/reflex-2/blob/main/notebooks/quickstart.ipynb)

## Results

Same inputs for both models: the same frames, the same 16 kHz audio. Reflex-2 uses heads trained on examples; Gemini 3.8 Flash is untrained, with thinking set to minimal (it cannot be turned off). Reflex-2 measured on one NVIDIA GB10 (ASUS Ascent GX10), Gemini via OpenRouter with network included, one clip at a time.

| Same 40 clips each | Gemini 3.8 Flash | **Reflex-2** |
|---|---|---|
| Falls vs daily activities | 85.0% | **97.5%** |
| Time per judgment (median) | 6.95 s | **0.28 s** |
| Surveillance: abnormal vs normal | 95% | 95% |
| Time per judgment (median) | 16.1 s | **1.4 s** |
| 10 sounds | 85% | **100%** |
| Time per judgment (median) | 3.66 s | **0.049 s** |

**Time from the fall to the alert** (watching the same clip like a camera)

| Scene | Fall starts | Gemini 3.8 Flash | **Reflex-2** |
|---|---|---|---|
| Falls from a chair | 3.4 s | 8.3 s after the fall | **1.9 s after the fall** |
| Walks and falls | 2.7 s | missed (all 4 answers "normal") | **2.1 s after the fall** |
| Falls forward | 2.6 s | missed (all 4 answers "normal") | **1.2 s after the fall** |
| Lies down on a bed (normal) | none | no alert | no alert |

## Training is what makes it work

Raw EmbeddingGemma 2 can only compare a clip with a text description, which is not enough for a judgment. Reflex-2 trains a head from examples. Accuracy on clips and sounds not used for training:

| Task | Examples | Training time | Raw EmbeddingGemma 2 | **Reflex-2** |
|---|---|---|---|---|
| Falls (people never seen in training) | about 120 | about 1 min | 53.8% | **95.7%** |
| Surveillance anomalies | 120 | about 3 min | 63.5% | **92.5%** |
| 10 sounds (ESC-50, official 5 folds) | 320 | about 30 s | 35% | **96.3%** |
| Voice emotion, 8 classes (speakers never seen) | 1,200 | about 1 min | 14% | **52%** |

Training time is embedding the examples on one GPU plus fitting the head (seconds on a CPU). Chance level: 50% for falls and surveillance, 10% for sounds, 12.5% for emotion.

## Heads

| File | Judgment | Labels | Training data | Accuracy (leave-one-person-out) |
|---|---|---|---|---|
| `fall.json` | a person falls, indoor home camera | `fall`, `adl` (daily activity) | GMDCSA24, 160 clips, 4 actors, MIT License | 95.7% balanced |

A head is a logistic regression over the 768-d embedding of a clip (frames at 1 per second, up to 32), stored as JSON (labels, normalization, weights). This is the standard linear-probe recipe; what is new is the base model, which reads video and audio and runs on phones. The surveillance head is not released, because UCF-Crime is for research use.

## Notes

- Judgments need a head trained on examples; raw EmbeddingGemma 2 does not judge reliably.
- Reflex-2 tells **when** something happens, not **who**. Combine it with a person detector to locate people.
- Evaluations use public data (staged falls, surveillance clips, sound effects). For your site, train the head on your own footage.

## License

Apache 2.0. Base model: Google EmbeddingGemma 2 (Apache 2.0); Reflex-2 is not affiliated with or endorsed by Google. `fall.json` is trained on GMDCSA24 (MIT License). Comparison data: UCF-Crime (Sultani et al. 2018), ESC-50 (Piczak 2015, CC BY-NC 3.0), RAVDESS (Livingstone and Russo 2018, CC BY-NC-SA 4.0). Gemini 3.8 Flash results were measured by the author through OpenRouter. Concept: NerveReflex.
