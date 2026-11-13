# Вимірювання на SparkFun Edge для статті AIR'2026 — покрокова інструкція

Мета — заповнити табл. «Measurements on the SparkFun Edge» (Flash, Arena, Latency, Energy, I_avg)
і речення в анотації для **моделі статті** (int8 QAT DS-1D-CNN, 60,0 % на тесті етилен+CO).

На фото: **SparkFun Edge** (Ambiq Apollo3 Blue, без власного стабілізатора — живлення лише 3,0–3,6 В),
**Serial Basic Breakout USB-C** (мікросхема CH340C) і камера HM01B0 (для статті не потрібна).

## Що змінено в цьому архіві відносно репозиторію

| Що | Навіщо |
|---|---|
| `firmware/src/main.cc`, `hal_apollo3.c`, `hal.h` | нові команди: `S` — deep sleep (для I_sleep), `D` — режим «сон → виведення раз на 1 с» з пробудженням від STIMER (I_avg **вимірюється**, а не лише рахується за формулою) |
| `python/hil_host.py` | ключі `--duty`, `--sleep`, `--i-idle-ma`, `--i-duty-ua`, `--text/--data`; довгі бенчмарки більше не обриваються через 10-секундний тайм-аут; повторне підключення після скидання плати |
| `firmware/gcc/Makefile` | `TFLM_KERNELS=cmsis_nn` або `reference`: раніше при двох збірках TFLM Makefile міг тихо взяти бібліотеку CMSIS-NN і для рядка float32 |
| `variants/qat_int8`, `variants/float32`, `python/use_variant.py` | готові файли прошивки, нічого не треба перенавчати чи експортувати |
| `build_all.sh` (WSL), `flash.cmd` (Windows) | збирання трьох прошивок і завантаження однією командою |
| `python/extra_real.py`, `extra_methane.py`, `train.py` | скрипти, на які посилається стаття і відповідь рецензенту R2 |

**Важливо про моделі.** У попередньому архіві `results/ds_cnn_qat_int8.tflite` був з *іншого* прогону
навчання (52,6 %), а модель статті (60,0 %) містилася лише у `firmware/src/model_data.cc`. Тому
**не запускайте `export_firmware.py`**: використовуйте `use_variant.py` (це робить `build_all.sh`).
Float32-модель статті (55,9 %) не збереглася, тому `variants/float32` — та сама архітектура з іншого
прогону. Для рядка float32 вона придатна **лише для вартості** (Flash, арена, затримка, енергія),
бо ці величини від значень ваг не залежать. Точність float32 з плати в статтю не пишіть.

Перевірено тут: протокол `hil_host.py` (на імітаторі плати), C-код HAL — синтаксичний аналіз clang
для Cortex-M4 проти заголовків AmbiqSuite SDK і BSP SparkFun Edge (0 попереджень з `-Wall`),
`main.cc` — проти заглушок API TFLM. **Повна збірка й запуск на платі ще не виконувалися — це робите ви.**

---

## Крок 0. Що потрібно

- кабель USB-C **з даними** (не лише зарядний);
- для енергії: **Nordic PPK2** або **Joulescope** (найкраще). Без них — мультиметр (див. крок 9, варіант Б);
- 5–6 перемичок «мама–мама»;
- батарейку CR2032 з тримача на звороті **вийняти**.

## Крок 1. Залізо і драйвер (Windows)

1. На звороті Serial Basic перевірте перемичку напруги: має бути **3.3V** (5 В пошкодить Edge).
   Для певності виміряйте мультиметром VCC–GND на роз'ємі Serial Basic: ≈ 3,3 В.
2. Від'єднайте шлейф камери (відкрийте фіксатор роз'єму CAMERA). Камера не потрібна і лише додає струм.
3. Вставте Serial Basic у 6-контактний роз'єм Edge так, як на фото (написи BLK…GRN збігаються).
4. Під'єднайте USB-C до ПК. *Диспетчер пристроїв → Порти (COM і LPT)* → має з'явитися
   **USB-SERIAL CH340 (COMx)**. Запам'ятайте номер, далі — `COM8` як приклад.
   Якщо пристрій без драйвера («Невідомий пристрій»): встановіть драйвер CH340 з інструкції SparkFun
   «How to Install CH340 Drivers» (файл виробника WCH `CH341SER.EXE`), перепід'єднайте кабель.

## Крок 2. Програми

**IDE.** Окремої IDE не потрібно: прошивка збирається `make`. Для редагування коду зручно взяти
**VS Code** з розширеннями *WSL* і *C/C++* (відкривати папку командою `code .` з терміналу WSL).
Arduino IDE не використовуйте: там інше ядро Apollo3 і інша версія TFLM, числа не збігатимуться зі статтею.

**WSL (Linux усередині Windows)** — для збирання. PowerShell *від імені адміністратора*:
```powershell
wsl --install -d Ubuntu-22.04
```
Перезавантажте ПК, відкрийте «Ubuntu 22.04», створіть користувача. Далі в Ubuntu:
```bash
sudo apt update
sudo apt install -y build-essential git make python3 python3-pip python3-numpy python3-pil unzip wget curl
```

**Python у Windows** — для завантаження прошивки і HIL (COM-порт доступний лише з Windows).
Python 3.11 з python.org, потім у `cmd`:
```cmd
py -3.11 -m venv C:\venvs\hil
C:\venvs\hil\Scripts\activate
python -m pip install numpy pyserial pycryptodome
```

## Крок 3. Розпакувати архів

Розпакуйте `tinyml_hil_kit.zip` у `C:\work\` → вийде `C:\work\tinyml-gas-mixture-apollo3`.
У WSL ця ж папка — `/mnt/c/work/tinyml-gas-mixture-apollo3`.
(Після вимірювань закомітьте вміст у GitHub-репозиторій `akakiev/tinyml-gas-mixture-apollo3`.)

## Крок 4. AmbiqSuite SDK і дві бібліотеки TFLM (WSL, ≈ 20–40 хв, потрібен інтернет)

```bash
cd ~
git clone --recursive https://github.com/sparkfun/AmbiqSuiteSDK
git clone https://github.com/tensorflow/tflite-micro
cd tflite-micro
# 1) з CMSIS-NN — для рядків int8 (заодно завантажить ARM GCC у downloads/gcc_embedded)
make -f tensorflow/lite/micro/tools/make/Makefile -j8 TARGET=cortex_m_generic \
     TARGET_ARCH=cortex-m4+fp OPTIMIZED_KERNEL_DIR=cmsis_nn BUILD_TYPE=release microlite
# 2) без CMSIS-NN (референсні ядра) — для рядка float32
make -f tensorflow/lite/micro/tools/make/Makefile -j8 TARGET=cortex_m_generic \
     TARGET_ARCH=cortex-m4+fp BUILD_TYPE=release microlite
ls gen/   # мають бути ..._release_cmsis_nn_gcc і ..._release_gcc
```

## Крок 5. Зібрати три прошивки (WSL)

```bash
cd /mnt/c/work/tinyml-gas-mixture-apollo3
chmod +x build_all.sh
./build_all.sh
```
Результат у `builds/`:

| Файл | Рядок таблиці |
|---|---|
| `A_float32_48MHz.bin` | float32, no CMSIS-NN, 48 MHz (арена 64 КБ, щоб float-модель точно вмістилася) |
| `B_int8_48MHz.bin` | int8 QAT, CMSIS-NN, 48 MHz |
| `C_int8_96MHz.bin` | int8 QAT, CMSIS-NN, 96 MHz (TurboSPOT) |

Скрипт друкує рядки `arm-none-eabi-size` (**Flash = text + data**) і `builds/versions.txt`
(коміти TFLM і SDK, версія GCC — для відтворюваності, варто згадати в статті).
В усіх прошивках GPIO **36** (контакт «36» на нижній гребінці) тримається в «1» під час виведення —
це мітка для логічного входу профайлера.

## Крок 6. Завантажити прошивку (Windows, `cmd`)

Serial Basic вставлено в Edge напряму (як на фото), USB під'єднано.
```cmd
C:\venvs\hil\Scripts\activate
cd C:\work\tinyml-gas-mixture-apollo3
```
1. Натисніть і **тримайте кнопку 14**.
2. Не відпускаючи 14, натисніть і відпустіть **RST**.
3. Запустіть: `flash.cmd COM8 builds\B_int8_48MHz.bin`
4. Коли пішла передача пакетів — відпустіть 14. Наприкінці скрипт сам скине плату.
   Якщо після завантаження плата мовчить — натисніть RST (без 14).

Перевірка (≈ 30 с):
```cmd
python python\hil_host.py --port COM8 --npz variants\qat_int8\hil_test.npz --n 20 --bench 100 --out results\smoke.json
```
Має надрукувати `Пристрій: {... 'cpu_hz': '48000000', 'in_type': 'int8' ...}` і JSON з
`class_agreement_with_pc_pct: 100.0`.

## Крок 7. Точність і затримка (усі три прошивки, живлення від Serial Basic)

Для кожної прошивки: завантажити (крок 6), потім:

```cmd
rem B: int8, 48 МГц
python python\hil_host.py --port COM8 --npz variants\qat_int8\hil_test.npz --n 500 --bench 1000 --label "int8 48MHz" --out results\hil_int8_48MHz.json
rem C: int8, 96 МГц  (у виводі має бути cpu_hz=96000000)
python python\hil_host.py --port COM8 --npz variants\qat_int8\hil_test.npz --n 500 --bench 1000 --label "int8 96MHz" --out results\hil_int8_96MHz.json
rem A: float32, 48 МГц  (лише вартість)
python python\hil_host.py --port COM8 --npz variants\float32\hil_test.npz --n 200 --bench 300 --label "float32 48MHz" --out results\hil_float32.json
```
500 вікон по 6,4 КБ на 115 200 бод — приблизно 5–6 хв.

Критерії: `class_agreement_with_pc_pct` ≥ 99,5 % (очікується 100 %), `max_abs_diff_int8_*` ≤ 2;
для B і C `pc_accuracy_pct_same_windows` ≈ `mcu_accuracy_pct` (≈ 60 %, бо це той самий тест).
У JSON: `bench_inference_ms` — **Latency**, `device.arena_used` — **Arena**,
`preprocess_ms_mean` — час передобробки (згадаємо в тексті).

## Крок 8. Схема для вимірювання енергії

Serial Basic **не повинен живити** плату під час вимірювань, інакше струм іде в обхід профайлера.

1. Вийміть Serial Basic з роз'єму. З'єднайте його з Edge перемичками **позиція в позицію**
   (як були б з'єднані напряму) для **GND, TXO, RXI, DTR**; **VCC і CTS не з'єднувати**.
2. PPK2 у режимі **Source meter**, 3300 мВ: `VOUT` → контакт **VDD** нижньої гребінки Edge,
   `GND` → **GND** цієї ж гребінки.
3. Логічний порт PPK2: `VCC` → VDD плати (опорна напруга), `GND` → GND, `D0` → контакт **36**.
4. Порядок: спочатку ввімкнути живлення PPK2, потім під'єднати USB Serial Basic
   (інакше плата частково живиться через лінію TX — спотворює сон).
5. Завантаження прошивки (крок 6) працює і в цій схемі: GND/TX/RX/DTR під'єднані.

Joulescope — аналогічно (у розрив VDD, живлення 3,3 В від нього або від лабораторного блока).

## Крок 9. Послідовність вимірювань (для кожної прошивки)

PPK2: частота дискретизації 100 kS/s; середнє й заряд читати у виділеній ділянці (Selection).
Номер N для бенчмарку беріть так, щоб він тривав ≥ 15 с: `N ≈ 15000 / bench_inference_ms` (з кроку 7).

| # | Що | Команда (`cmd`) | Що зчитати з PPK2 |
|---|---|---|---|
| a | **I_idle** — CPU активне, чекає команду, без мережі | натиснути RST, нічого не запускати | середній струм за 10 с, мА |
| b | **I_act** — безперервне виведення | `python python\hil_host.py --port COM8 --npz variants\qat_int8\hil_test.npz --n 0 --bench N --out results\bench_B.json` | середній струм на ділянці, де D0 = 1, мА |
| c | **I_avg** — 1 виведення/с, між ними deep sleep | `python python\hil_host.py --port COM8 --npz variants\qat_int8\hil_test.npz --n 0 --bench 0 --duty 70 --out results\duty_B.json` | виділити рівно 60 періодів (від фронту D0 до фронту через 60 імпульсів): середній струм, мкА; і заряд одного імпульсу Q, мкКл, та його тривалість |
| d | **I_sleep** — deep sleep | `python python\hil_host.py --port COM8 --npz variants\qat_int8\hil_test.npz --n 0 --bench 0 --sleep` | середнє за ≥ 30 с, мкА; потім Enter |

Для прошивки C — те саме (a–d); перевірте, що в `duty_C.json` поле `duty_work_ms` менше, ніж у B
(тобто після сну ядро справді повертається на 96 МГц). Для A — достатньо a і b (`--npz variants\float32\hil_test.npz`).

Підсумковий розрахунок (приклад для B; підставте свої числа, text/data — з `builds\B_int8_48MHz.size`):
```cmd
python python\hil_host.py --port COM8 --npz variants\qat_int8\hil_test.npz --n 0 --bench 2000 ^
  --v 3.3 --i-active-ma <I_act> --i-idle-ma <I_idle> --i-sleep-ua <I_sleep> --i-duty-ua <I_avg> ^
  --text <text> --data <data> --label "int8 48MHz" --out results\energy_int8_48MHz.json
```
Отримаєте `energy_uj_per_inference_gross` (V·I_act·t), `energy_uj_per_inference_net`
(V·(I_act − I_idle)·t — внесок саме мережі), `avg_current_ua_1hz_model` (за формулою) і
`avg_current_ua_1hz_measured` (виміряно в кроці c). Розбіжність двох останніх ≤ 10–15 % — ознака, що
вимірювання узгоджені. Перехресна перевірка: V·Q одного імпульсу ≈ енергія передобробки + виведення.

**Варіант Б — без PPK2/Joulescope.** Мультиметр у розрив VDD (схема кроку 8, живлення 3,3 В від
лабораторного блока): I_idle, I_act (довгий `--bench`, показ стабільний) і I_sleep (діапазон мкА)
вимірюються коректно. I_avg у режимі `--duty` мультиметр не усереднює — тоді в таблицю йде лише
`avg_current_ua_1hz_model`, і в підписі таблиці так і пишемо.

## Крок 10. Що надіслати мені

- `results\hil_int8_48MHz.json`, `hil_int8_96MHz.json`, `hil_float32.json`;
- `results\energy_*.json`, `duty_*.json`;
- `builds\*.size` і `builds\versions.txt`;
- модель профайлера, напругу, частоту дискретизації; бажано скріншот PPK2 з 2–3 імпульсами режиму `--duty`
  (можна дати як рисунок, якщо дозволить обсяг).

Я впишу числа в табл. «Measurements on the SparkFun Edge» і анотацію, а в підписі таблиці уточню:
енергія «net» = V·(I_act − I_idle)·t, I_avg виміряно з пробудженням від STIMER, вимірювання на рівні
плати (з мікрофонами, акселерометром тощо), без нагрівачів MOX.

## Якщо щось не так

| Симптом | Причина / що зробити |
|---|---|
| немає COM-порту | кабель без даних або немає драйвера CH340 (крок 1) |
| `flash.cmd`: `Failed to ... / Tries = 3` | не той порт; знову: тримати 14 → RST → запустити → відпустити 14 |
| `E,setup,3` | замала арена: перезберіть з більшою `ARENA_SIZE` (у `build_all.sh`) |
| `E,setup,2` | несумісна схема TFLite: модель і TFLM різних версій — повідомте мені версію TFLM |
| `Плата не відповідає` у `hil_host.py` | перехрещені TX/RX у схемі кроку 8; прошивку не запущено (натисніть RST без 14); спробуйте `--boot-wait 3` |
| у прошивці C `cpu_hz=48000000` | TurboSPOT не ввімкнувся: перевірте, що збирали з `-DUSE_BURST_MODE` (`builds\C_int8_96MHz.log`) |
| `class_agreement` < 99 % для int8 | у `firmware/src` не та модель: `python3 python/use_variant.py qat_int8` і перезбірка |
| I_sleep сотні мкА | під'єднана камера; живлення через Serial Basic VCC; USB під'єднано до ввімкнення PPK2; батарейка в тримачі |
