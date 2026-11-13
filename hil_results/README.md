# HIL results on the SparkFun Edge (Apollo3 Blue), 13 Nov 2026

Measured with `python/hil_host.py` over UART (115200 baud), test windows from `variants/<variant>/hil_test.npz`
(ethylene + CO, chronological test split). Table 4 of the paper is built from these files.

| File | Firmware | Windows | Benchmark |
|---|---|---|---|
| `hil_float32.json` | A: float32, reference kernels, 48 MHz (cost-only model, see `variants/README.md`) | 200 | 300 inferences |
| `hil_int8_48MHz.json` | B: int8 QAT (paper model), CMSIS-NN, 48 MHz | 500 | 1000 inferences |
| `hil_int8_96MHz.json` | C: int8 QAT (paper model), CMSIS-NN, 96 MHz TurboSPOT | 500 | 1000 inferences |

`*.size` — `arm-none-eabi-size` of each image (Flash = text + data). Firmware images (md5):
A `166ae84acb6009db683ff9dba68e1a5d`, B `111795c571575ce8594fd7098624b389`, C `479cf19d32fd38101d6614fda23d53ba`.

Toolchain:
```
tflite-micro: b8d75cbf6b8009762bc1e3ee4817bc33e5dcee47
AmbiqSuiteSDK: e280cbde3e366509da6768ab95471782a05d2371
gcc: arm-none-eabi-gcc (Arm GNU Toolchain 14.3.Rel1 (Build arm-14.174)) 14.3.1 20250623
```

`device_table_estimates.csv` — Table 4. Flash, arena and latency are measured. Energy per inference and the
average current at one inference per second are **estimates**: measured latency x typical currents from the
Apollo3 Blue SoC Datasheet DS-A3-1p2p0 (2022), Table 1175, at 3.3 V (CoreMark 10.3 uA/MHz at 48 MHz,
27 uA/MHz at 96 MHz; deep sleep with 384 kB retention 4.1 uA). MCU only; board peripherals and MOX heaters
are not included. No current measurement was made (the firmware's `D`/`S` modes are ready for one).
