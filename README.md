# Whisper + Diarization Interface

This application utilizes OpenAI's Whisper and PyAnnote' Segmentation-3.0 as well as Speaker-Diarization-3.1 to create an automatic subtitle creator.

![header2.png](assets/header2.png)

## Prerequisites

You must have docker with GPU access enabled.

+ Accept [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0) user conditions
+ Accept [pyannote/speaker-diarization-3.1](https://hf.co/pyannote-speaker-diarization-3.1)user conditions
+ Create access token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens).


## Installation

To install the demo

```bash
git clone https://github.com/tdolan21/whisper-diarization-subtitles
cd whisper-diarization-subtitles
pip install -r requirements.txt
```
then you can add you HuggingFace token to the .env

```env
HUGGINGFACE_HUB_TOKEN=your_huggingface_hub_token
```
or you can install with docker for quick usage:

```bash
docker compose up --build
docker compose down -v
```


## Citations

```bibtex
@misc{radford2022whisper,
  doi = {10.48550/ARXIV.2212.04356},
  url = {https://arxiv.org/abs/2212.04356},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

```bibtex
@inproceedings{Plaquet23,
  author={Alexis Plaquet and Hervé Bredin},
  title={{Powerset multi-class cross entropy loss for neural speaker diarization}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}
```
```bibtex
@inproceedings{Bredin23,
  author={Hervé Bredin},
  title={{pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}

```
