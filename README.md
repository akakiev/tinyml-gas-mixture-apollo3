# TinyML-розпізнавання складу газових сумішей на Ambiq Apollo3

*TinyML recognition of gas-mixture composition and concentration estimation with a multi-task
depthwise-separable 1D-CNN (4.4 k parameters, int8) on the Ambiq Apollo3 Blue microcontroller
(SparkFun Edge): training, PTQ/QAT quantization, TensorFlow Lite Micro firmware and
hardware-in-the-loop evaluation. Documentation is in Ukrainian.*

Код для експериментів до статті «Розпізнавання складу газових сумішей на основі TinyML на
ультранизькоенергетичному мікроконтролері Ambiq Apollo3»: навчання DS-1D-CNN, квантизація,
прошивка для SparkFun Edge і вимірювання в режимі hardware-in-the-loop (HIL).

```
python/
  common.py           дані UCI, передобробка, вікна, розбиття, модель, метрики
  train.py            повний експеримент -> табл. 2 + експорт моделі в прошивку
  export_firmware.py  перемкнути модель у прошивці (float32 / ptq_int8 / qat_int8) без навчання
  hil_host.py         HIL-вимірювання на платі -> табл. 3
firmware/
  src/main.cc         TFLM-інференс, передобробка, протокол UART
  src/hal_apollo3.c   UART, лічильник тактів DWT, TurboSPOT, GPIO-тригер (AmbiqSuite)
  src/model_data.cc   модель (генерується; у репозиторії — заглушка)
  src/model_config.h  статистики нормування (генерується)
  gcc/Makefile        збірка для SparkFun Edge + bootload_asb / bootload_svl
  host/               та сама прошивка як програма для ПК (перевірка без плати)
```

## Що вже перевірено, а що ні

Перевірено на вашому комп'ютері (Linux-середовище Claude, TF 2.15.1, tfmot 0.8.0), на штучних даних:

| Частина | Результат |
|---|---|
| `train.py --synthetic --epochs 15 --qat-epochs 3` — весь конвеєр | працює: дані → базові моделі → CNN → PTQ → QAT → TFLite → експорт у прошивку |
| Якість на штучних даних (лише перевірка, НЕ для статті) | CNN float32 93,7 %, int8 PTQ 93,7 %, int8 QAT 94,0 % |
| Прошивка `main.cc` + TFLM (збірка для ПК) з експортованою моделлю | збіг класу з еталоном TFLite — **100 %** (521 вікно); різниця сирих int8-виходів ≤ 2 LSB |
| Арена TFLM | 6 912 Б на x86-64 (на Cortex-M — менше; беріть значення з плати) |
| Модель | 4 374 параметри, 186 816 MAC; файл .tflite int8 ≈ 14 КБ (ваги ≈ 4,4 КБ + метадані, масштаби, int32-зміщення) |
| `hal_apollo3.c` | синтаксис перевірено проти заголовків AmbiqSuite SDK 2.4 |
| Крос-компіляція і запуск на SparkFun Edge | **ще не виконувались** |

Набір UCI з того середовища недоступний (мережевий фільтр), тому справжній експеримент
запускайте у Windows (крок 2) — там обмежень немає, і він швидший.

Нюанс TensorFlow 2.15: **не встановлюйте `tf_keras`** — з ним QAT падає («not a supported
layer type»). `common.setup_keras_env()` сам вибирає правильний Keras для TF 2.15 і 2.16+.

## 1. Python-середовище

Потрібен Python 3.10 або 3.11 (TensorFlow 2.15 не підтримує новіші). У Windows (`cmd`)
створюйте середовище з **коротким шляхом**, інакше `pip` може впасти з `WinError 206`
(задовгий шлях, особливо з Python із Microsoft Store). Команди — по одній:

```cmd
python -m venv C:\venvs\tg
C:\venvs\tg\Scripts\activate
python -m pip install -r requirements.txt
```

Linux/WSL: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

## 2. Навчання й табл. 2

```bash
cd python
python train.py --synthetic --epochs 15 --qat-epochs 3  # ~3 хв: перевірка, що все встановлено
python train.py                                         # основний експеримент (етилен + CO)
python train.py --recording methane --no-export         # узагальнення (етилен + метан)
```

Скрипт сам завантажить набір UCI (≈ 250 МБ). Якщо завантаження не працює, візьміть архів зі
сторінки https://archive.ics.uci.edu/dataset/322 і розпакуйте `ethylene_CO.txt` та
`ethylene_methane.txt` у папку `data/`.

Результати в `results/`:
- `table2_co.md` — готові рядки табл. 2 (accuracy, macro-F1, MAE, R², розмір);
- `results_co.json` — усе, включно з матрицями плутанини (для тексту про помилки на перехідних ділянках);
- `ds_cnn_*.tflite`, `hil_test.npz`, `preproc.npz`.

Прошивка автоматично отримує кращу з int8-моделей (PTQ чи QAT за macro-F1 на валідації).

## 3. Бібліотека TFLM для Cortex-M4F (Linux або WSL)

```bash
git clone https://github.com/tensorflow/tflite-micro && cd tflite-micro
make -f tensorflow/lite/micro/tools/make/Makefile -j8 \
     TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+fp OPTIMIZED_KERNEL_DIR=cmsis_nn \
     BUILD_TYPE=release microlite
```

Make сам завантажить ARM GCC у `tensorflow/lite/micro/tools/make/downloads/gcc_embedded/`.
**Прошивку збирайте тим самим компілятором**, що й бібліотеку (див. `TOOLCHAIN` нижче).
Для рядка «float32 без CMSIS-NN» у табл. 3 зберіть другу бібліотеку без `OPTIMIZED_KERNEL_DIR`.

## 4. Прошивка для SparkFun Edge

```bash
git clone --recursive https://github.com/sparkfun/AmbiqSuiteSDK ~/AmbiqSuiteSDK
cd firmware/gcc
TC=~/tflite-micro/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin/arm-none-eabi
make SDKPATH=~/AmbiqSuiteSDK TFLM_DIR=~/tflite-micro TOOLCHAIN=$TC
make SDKPATH=~/AmbiqSuiteSDK TFLM_DIR=~/tflite-micro TOOLCHAIN=$TC bootload_asb COM_PORT=/dev/ttyUSB0
```

У Windows зручно збирати у WSL, а завантажувати з Windows тією самою командою `asb.py`, яку
друкує `make bootload_asb` (з `-port COM8`). Перед завантаженням через ASB утримуйте кнопку
14 і натисніть Reset.

`make` друкує розмір образу (`arm-none-eabi-size`): **Flash = text + data**.
Опції: `EXTRA_DEFS="-DUSE_BURST_MODE"` (96 МГц), `EXTRA_DEFS="-DPROFILE_PIN=<GPIO>"`,
`ARENA_SIZE=...`, `HIL_BAUD=...`.

Без плати ту саму прошивку можна перевірити на ПК:

```bash
cd tflite-micro && make -f tensorflow/lite/micro/tools/make/Makefile BUILD_TYPE=release microlite
cd firmware/host && make TFLM_DIR=~/tflite-micro selftest       # модель-тест із випадковими вагами
make TFLM_DIR=~/tflite-micro && python ../../python/hil_host.py --exec ./gas_host --n 100
```

## 5. HIL-вимірювання й табл. 3
> **Оновлення (листопад 2026).** Покрокова інструкція для SparkFun Edge, енергетичні режими прошивки
> (deep sleep, «1 виведення/с»), готові варіанти моделі статті та скрипти `build_all.sh` / `flash.cmd` —
> у [HIL_KIT_README.md](HIL_KIT_README.md). Модель для вимірювань перемикайте `python/use_variant.py`,
> а не `export_firmware.py`.


```bash
cd python
python hil_host.py --port COM8 --n 500 --bench 1000
```

Що звіряти:
- `class_agreement_with_pc_pct` має бути ≈ 100 %, а `max_abs_diff_int8_*` — 0–2
  (невеликі розбіжності дає округлення `log1pf` у передобробці);
- `mcu_mae_ppm` може бути трохи кращим за MAE з `table2`, бо прошивка обрізає від'ємні
  концентрації до 0 — так і опишіть у тексті;
- `bench_inference_ms` — затримка; `arena_used` — RAM-арена (стовпець «RAM (arena)»);
- `preprocess_ms_mean` — час передобробки (варто згадати в тексті окремо).

**Енергія.** Під'єднайте Nordic PPK2 / Joulescope в розрив живлення плати (бажано вимкнути
світлодіод живлення), запустіть `--bench 30000` (кілька секунд стабільного навантаження),
зафіксуйте середній струм `I_act`, потім струм у глибокому сні `I_sleep`, і передайте їх:

```bash
python hil_host.py --port COM8 --n 0 --bench 30000 --v 3.3 --i-active-ma <I_act> --i-sleep-ua <I_sleep>
```

Скрипт обчислить `energy_uj_per_inference` = V·I_act·t і `avg_current_ua_1hz` для режиму
«1 виведення/с». Зауважте, що I_act містить і фонове споживання плати (USB-UART тощо); для
чистішої оцінки віднімайте струм тієї ж плати в активному циклі очікування.

Рядки табл. 3:
1. `python export_firmware.py --variant float32`, прошивка з бібліотекою без CMSIS-NN → рядок float32;
2. `python export_firmware.py --variant qat_int8` (або `ptq_int8`), бібліотека з CMSIS-NN → рядок int8;
3. те саме з `EXTRA_DEFS="-DUSE_BURST_MODE"` → рядок 96 МГц.

## Що оновити в тексті статті

- **Децимація:** у коді — усереднення за блоками по 10 відліків (найпростіший антиаліасинговий
  фільтр). Так і напишіть у п. 3.4 або замініть фільтр у `common.load_recording`.
- **Softmax:** модель видає логіти, softmax обчислюється після деквантизації (для класу достатньо
  argmax). Рис. 2 лишається коректним за змістом.
- **RAM:** у п. 3.5 оцінка 3,2 КБ стосується лише активацій; у табл. 3 вписуйте `arena_used`
  з плати — вона більша, бо включає службові структури TFLM.
- **1D-згортки** реалізовано як Conv2D/DepthwiseConv2D з ядром (k, 1). Це стандартний прийом:
  саме так їх виконують оптимізовані ядра CMSIS-NN. Можна згадати одним реченням у п. 3.6.
