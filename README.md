# TinyML gas-mixture recognition on the Ambiq Apollo3 Blue (SparkFun Edge)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24207494.svg)](https://doi.org/10.5281/zenodo.24207494)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code and on-board results for the paper

> S. Baraban, O. Zhahlovska. *TinyML Gas-Mixture Recognition on an Ultra-Low-Power Microcontroller:
> Lessons from Simulation and Leakage-Free Real-Data Evaluation.* Artificial Intelligence Research
> (AIR'2026), American University in the Emirates, Dubai (Springer proceedings, in press).

The repository contains the full pipeline: data preparation for the public UCI *Gas sensor array under
dynamic gas mixtures* recordings, training of a multi-task depthwise-separable 1D-CNN and of feature
baselines, int8 quantization (PTQ and QAT), TensorFlow Lite Micro (TFLM) firmware for the SparkFun Edge
(Apollo3 Blue, Cortex-M4F) and a hardware-in-the-loop (HIL) protocol that streams test windows to the board
over UART and compares its raw int8 outputs with the PC interpreter.

## Main results

Real data, chronological 70/15/15 split, 10-s windows, seed 42 (Table 2 of the paper):

| Model | Size | Ethylene + CO accuracy | Ethylene + CH4 accuracy |
|---|---|---|---|
| DS-1D-CNN, float32 / int8 QAT (4,374 parameters, 186,816 MACs) | 14.0 KB (int8) | 55.9 % / 60.0 % | 82.5 % |
| int8 MLP on 48 window statistics | 8.1 KB | **85.9 %** | **85.8 %** |

A random split of overlapping windows inflates the CNN's accuracy by 19 percentage points
(86.9 % vs 67.6 % with identical training settings).

On the SparkFun Edge, measured on 13 Nov 2026 ([`hil_results/`](hil_results/), Table 4 of the paper):

| Firmware | Flash | Arena | Latency | Energy* | I_avg* (1 inference/s) |
|---|---|---|---|---|---|
| float32, reference kernels, 48 MHz | 72.9 KB | 15.0 KB | 86.88 ms | 141.7 uJ | 49.6 uA |
| int8 QAT, CMSIS-NN, 48 MHz | 105.4 KB | 6.1 KB | 23.43 ms | 38.2 uJ | 19.0 uA |
| int8 QAT, CMSIS-NN, 96 MHz (TurboSPOT) | 105.8 KB | 6.1 KB | 11.76 ms | 100.6 uJ | 43.6 uA |

The int8 firmware agrees with the PC TFLite interpreter in 100 % of class decisions on 500 test windows
(raw outputs within 1 LSB). \*Energy and average current are **estimates** from the measured latency and typical
datasheet currents (Apollo3 Blue SoC Datasheet DS-A3-1p2p0, Table 1175, 3.3 V); MCU only, sensor heaters
excluded. The float32 row uses a network of the same architecture from another training run and is reported
for cost only (see [`variants/README.md`](variants/README.md)).

## Repository layout

```
python/
  common.py            UCI data loading, decimation, windows, chronological split, models, metrics
  train.py             main experiment (Table 2) and export of the selected model to the firmware
  extra_real.py        size-constrained random forest, int8 MLP, chronological vs random split (ethylene + CO)
  extra_methane.py     the same baselines on the ethylene + methane recording
  hil_host.py          HIL host: accuracy check, latency benchmark, duty-cycle and deep-sleep modes
  use_variant.py       switch the firmware model between the prepared variants (no retraining)
  export_firmware.py   export a .tflite from results/ into the firmware (not used for the paper numbers)
firmware/
  src/main.cc          TFLM inference, on-device preprocessing, UART protocol
  src/hal_apollo3.c    UART, DWT cycle counter, TurboSPOT, deep sleep, STIMER duty cycle, profiler GPIO
  gcc/Makefile         build for the SparkFun Edge (AmbiqSuite SDK)
  host/                the same firmware built as a PC program (verification without the board)
variants/              firmware model files: qat_int8 (the paper model) and float32 (cost only)
results/               model files (.tflite) and preprocessing statistics
hil_results/           raw on-board results behind Table 4, image sizes and tool versions
build_all.sh           builds the three firmware images (WSL/Linux)
flash.cmd              flashes an image from Windows through the Ambiq secure bootloader
HIL_KIT_README.md      step-by-step guide for the board measurements (in Ukrainian)
```

## Reproducing the results

### 1. Python environment

Python 3.10 or 3.11 (TensorFlow 2.15 does not support newer versions).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Do not install `tf_keras` with TF 2.15: QAT then fails with "not a supported layer type".

### 2. Training and Table 2

```bash
python python/train.py --synthetic --epochs 15 --qat-epochs 3   # about 3 min: installation check
python python/train.py                                          # main experiment (ethylene + CO)
python python/train.py --recording methane --no-export          # second recording (ethylene + methane)
python python/extra_real.py                                     # baselines and split comparison (needs data/co_decim10.npz from train.py)
python python/extra_methane.py
```

`train.py` downloads the UCI data set (dataset 322, about 250 MB). If the download fails, take the archive from
https://archive.ics.uci.edu/dataset/322 and unpack `ethylene_CO.txt` and `ethylene_methane.txt` into `data/`.
Note that the CNN's accuracy on the drifting test period depends noticeably on the training schedule.

### 3. Firmware (Linux or WSL)

```bash
git clone --recursive https://github.com/sparkfun/AmbiqSuiteSDK ~/AmbiqSuiteSDK
git clone https://github.com/tensorflow/tflite-micro ~/tflite-micro && cd ~/tflite-micro
git checkout b8d75cbf   # commit used for the paper
make -f tensorflow/lite/micro/tools/make/Makefile -j8 TARGET=cortex_m_generic \
     TARGET_ARCH=cortex-m4+fp OPTIMIZED_KERNEL_DIR=cmsis_nn BUILD_TYPE=release microlite
make -f tensorflow/lite/micro/tools/make/Makefile -j8 TARGET=cortex_m_generic \
     TARGET_ARCH=cortex-m4+fp BUILD_TYPE=release microlite
cd <this repository> && bash build_all.sh   # builds/A_float32_48MHz.bin, B_int8_48MHz.bin, C_int8_96MHz.bin
```

The paper images were built with TFLM `b8d75cbf`, AmbiqSuiteSDK `e280cbde` and Arm GNU Toolchain 14.3.Rel1
([`hil_results/versions.txt`](hil_results/versions.txt)); their md5 sums are listed in
[`hil_results/README.md`](hil_results/README.md).

### 4. Flashing and HIL measurements (Windows)

```cmd
pip install numpy pyserial pycryptodome
rem hold button 14, press and release RST, then:
flash.cmd COM8 builds\B_int8_48MHz.bin
python python\hil_host.py --port COM8 --npz variants\qat_int8\hil_test.npz --n 500 --bench 1000 --out results\hil_int8_48MHz.json
```

`hil_host.py --duty N` runs N cycles of "deep sleep, wake on STIMER, preprocess + infer" with a 1-s period and
`--sleep` puts the board into deep sleep, so that the average and sleep currents can be measured with a power
profiler (GPIO 36 is high during inference). These modes were not used for the paper, because no current
measurement was made. Without a board, the firmware can be checked on a PC:

```bash
cd firmware/host && make TFLM_DIR=~/tflite-micro && python ../../python/hil_host.py --exec ./gas_host --n 100
```

## Data

UCI Machine Learning Repository, *Gas sensor array under dynamic gas mixtures* (J. Fonollosa, 2015),
https://archive.ics.uci.edu/dataset/322; see also Fonollosa et al., Sensors and Actuators B 215 (2015) 618-629,
https://doi.org/10.1016/j.snb.2015.03.028. The data are not redistributed here.

## Citation

If you use this code, please cite the paper above and this software archive (see [`CITATION.cff`](CITATION.cff)):

> Baraban, S., Zhahlovska, O. (2026). tinyml-gas-mixture-apollo3 (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.24207494

The DOI above identifies version 1.0.0, the version used in the paper; https://doi.org/10.5281/zenodo.24207493 always resolves to the latest version.

## Authors and maintenance

- Serhii Baraban, Poznan University of Technology, Poland ([ORCID 0000-0001-9535-1644](https://orcid.org/0000-0001-9535-1644)), serhii.baraban@put.poznan.pl
- Olena Zhahlovska, American University in the Emirates, Dubai, United Arab Emirates ([ORCID 0000-0001-9493-6798](https://orcid.org/0000-0001-9493-6798)), olena.zhahlovska@aue.ae

The repository is maintained by the first author under the GitHub account `akakiev`.
A Ukrainian version of this description is in [`README_uk.md`](README_uk.md).

## License

MIT, see [`LICENSE`](LICENSE).
