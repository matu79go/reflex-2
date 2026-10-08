# Reflex-2

**Fast, local judgments on video. 740M parameters, one GPU, no cloud.**

Reflex-2 judges camera video, images and audio on your own hardware. It turns a clip into one vector with Google [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) and reads that vector with a small head trained from your examples:

- **Fall detection**: 95.7% on people the head never saw, 0.3 s per clip on one GPU. A ready-made head ships in `heads/fall.json`.
- **Faster than a frontier LLM, at the same or better accuracy.** On the same clips, Gemini 3.8 Flash takes 7 to 16 s per clip through its API.
- **Your own judgment in seconds.** Give a few dozen labelled clips; `train_head()` fits a head in seconds on a CPU. The base model is never modified.
- **Video stays on the device.** Nothing is sent to a cloud service.

Reflex-2 is the video and audio sibling of [Reflex-1](https://github.com/matu79go/reflex-1) (text and images, answers a written question with no training). Use Reflex-1 when the judgment is described in words; use Reflex-2 when it is seen or heard, and you can show examples.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matu79go/reflex-2/blob/main/notebooks/quickstart.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Reflex--2-yellow)](https://huggingface.co/matu79go/Reflex-2)

## At a glance

<p>
<img src="media/chart_fall_en.png" width="49%" alt="Fall detection: accuracy x speed">
<img src="media/chart_ucf_en.png" width="49%" alt="Surveillance anomalies: accuracy x speed">
</p>

| Same 40 clips each | **Reflex-2** | Gemini 3.8 Flash |
|---|---|---|
| Fall detection, accuracy (balanced) | **97.5%** | 85.0% |
| Fall detection, missed falls | **0 of 20** | 6 of 20 |
| Fall detection, median time per clip | **0.28 s** | 6.95 s |
| Surveillance anomalies, accuracy (balanced) | 95% | 95% |
| Surveillance anomalies, median time per clip | **1.4 s** | 16.1 s |
| Cost per clip | **0 (local)** | $0.007 to $0.023 |
| Training needed | a few dozen examples, seconds | none |

Reflex-2 measured on one NVIDIA GB10 (ASUS Ascent GX10), one clip at a time. Gemini 3.8 Flash via OpenRouter, network included, thinking set to minimal (it cannot be turned off), given the same frames as images and asked for a 0 to 100 score. Reflex-2's fall numbers come from heads that never saw the person in the clip (leave-one-person-out).

## Demos

**Fall detection: Reflex-2 vs Gemini 3.8 Flash.** Both see the same frames; a tag appears the moment each answers ([full video](media/race_en.mp4))

![Fall detection race](media/race_en.gif)

**Watching a home camera.** Every 0.5 s Reflex-2 judges the last 4 s; the line is the fall probability and the shaded band is the true fall ([full video](media/fall_en.mp4))

![Fall detection timeline](media/fall_en.gif)

**Surveillance.** One normal and one abnormal clip from UCF-Crime (face pixelated), with both models' answers and times

![Surveillance example](media/surveillance_pair_en.png)

## Results

**Fall detection** (GMDCSA24: 4 actors in 3 homes, 79 falls and 81 daily activities; each person held out in turn)

| Examples per class used for training | Balanced accuracy | AUC |
|---|---|---|
| none (similarity to "a person falling down") | n/a | 0.836 |
| 5 | 85.6% | 0.939 |
| 20 | 93.3% | 0.984 |
| all (about 60 per class) | **95.7%** | **0.990** |

**Surveillance anomalies** (UCF-Crime, binary normal vs abnormal; 120 training clips, 110 evaluation clips): AUC 0.978, balanced accuracy 92.5%, 1.4 s per clip (median 75 s of video, 32 frames). The UCF-Crime head is not released, because the dataset is for research use.

**Speed.** One GPU (NVIDIA GB10): 0.30 s for a 7 s clip at 1 frame per second, 2.2 GB of GPU memory. CPU only (4 Arm Cortex-X925 cores): about 11 s per clip, or 7.5 s with `tokens_per_frame=70` and 4 frames (91.3% balanced accuracy). Real-time use needs a GPU or an NPU; phone NPU numbers are not measured yet.

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+, PyTorch 2.4+, sentence-transformers 6.1+. The first run downloads `google/embeddinggemma-2` (1.5 GB). A GPU is recommended; CPU works for offline jobs.

## Use

```python
from reflex2 import Reflex2

rf = Reflex2().load_heads("heads")              # heads/fall.json -> head "fall"
rf.decide("clip.mp4", "fall")
# {'choice': 'fall', 'confidence': 0.97, 'probs': {'adl': 0.03, 'fall': 0.97}, 'latency_ms': 290}

rf.watch("camera.mp4", "fall")                  # judge the last 4 s every 0.5 s
# [(1.0, {'adl': 0.96, 'fall': 0.04}), ..., (4.5, {'adl': 0.03, 'fall': 0.97}), ...]
```

**Train your own judgment** from labelled clips, images or audio files:

```python
head = rf.train_head({"fall": ["f1.mp4", "f2.mp4", ...], "normal": ["n1.mp4", "n2.mp4", ...]}, name="my_fall")
head.save("heads/my_fall.json")
```

or from the command line:

```bash
python scripts/train_head.py --out heads/my_head.json --label fall="clips/fall/*.mp4" --label normal="clips/normal/*.mp4"
python scripts/watch.py camera.mp4 --head heads/my_head.json --label fall
```

**Untrained questions.** `rf.zero_shot(x, {"kitchen": "a kitchen", "street": "a street"})` picks the closest description. It works for "what is this" questions; for reliable judgments, train a head.

**HTTP endpoint** (same request shape as Reflex-1 and Jev):

```bash
python -m reflex2.server --heads heads --port 8098
```

```json
POST /v1/decisions
{"video": "clip.mp4",
 "questions": {"fall": {"type": "choice", "head": "fall"},
               "room": {"type": "choice", "instructions": "Where is this?",
                        "criteria": {"bedroom": "a bedroom", "kitchen": "a kitchen"}}}}
-> {"answers": {"fall": {"choice": "fall", "confidence": 0.97, "probs": {...}},
                "room": {"choice": "bedroom", "confidence": 0.95, "probs": {...}}}, "latency_ms": 300}
```

`video`, `image` and `audio` take a local path or base64; `state` takes text. One embedding serves every question in a request.

## How it works

1. **Read.** Frames are sampled at 1 per second (up to 32, evenly over the clip) and EmbeddingGemma 2 turns the clip into one 768-d vector.
2. **Judge.** A head (logistic regression over the vector) returns the probability of each option. Heads are a few hundred kilobytes of JSON.
3. **Watch.** For a live camera, `watch()` judges the last 4 s (2 frames per second) every 0.5 s.

The base model's weights are not modified. One vector can feed many heads at once (fall, intrusion, smoke...), so adding a judgment costs almost nothing at inference time.

## Limits

- Reflex-2 tells **when** something happens, not **who**: it does not draw boxes around people.
- Heads are trained per judgment. Without a head, Reflex-2 only does similarity search (weak for judgments such as emotion or falls).
- It does not read or reason over text instructions. For judgments described in words, use [Reflex-1](https://github.com/matu79go/reflex-1).
- Evaluations use public datasets (staged falls, YouTube surveillance clips). Accuracy in your setting should be checked on your own footage.

## License and credits

Code: Apache 2.0. Base model: Google EmbeddingGemma 2 (Apache 2.0); Reflex-2 is not affiliated with or endorsed by Google. The fall head is trained on GMDCSA24 (Data in Brief, 2024; MIT License). Comparison data: UCF-Crime (Sultani et al. 2018). Gemini 3.8 Flash results were measured by the author through OpenRouter. See `NOTICE`.

Blog post (Japanese and English): suzuki-shoten.dev/jp/projects/reflex-2 (coming soon). Concept: NerveReflex.
