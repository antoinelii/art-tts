# Articulatory-based Text-to-Speech (Art-TTS)

This project explores an alternative approach to text-to-speech (TTS) synthesis by generating speech through articulatory representations rather than traditional acoustic features like mel-spectrograms. By modeling how the vocal tract shapes sound production, we aim to create more controllable and interpretable speech synthesis systems.

It relies mainly on the combination of two previous papers, the idea being to:
- Adapt the generative power of a diffusion-based text-to-speech model (Grad-TTS) into an articulatory trajectories generator
- Use the Speech Articulatory Coding (SPARC) model decoding part to generate final speech using its articulatory based HiFi-GAN. Moreover its encoding part can be used as an AAI (Acoustic-to-Articulatory Inversion) to create articulatory pseudo-targets from speech data.


## Installation

### Installations to reuse Grad-TTS

Firstly, install all Python package requirements:

```bash
pip install -r requirements.txt
```

Secondly, build `monotonic_align` code (Cython):

```bash
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

TO BE COMPLETED

## Internship Report

The complete internship report is available in PDF format:

📘 **Internship Report (PDF)**  
 [Access the report](docs/Internship_Report.pdf)

## References

- SPARC https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding

- Grad-TTS https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS

## 🧑‍🔬 Supervision

This project was developed under the supervision of Emmanuel Dupoux and Angelo Ortiz Tandazo within the Cognitive Machine Learning (CoML) team at École Normale Supérieure (ENS) Ulm, Paris.