"""
Спільні функції: завантаження набору UCI «Gas sensor array under dynamic gas mixtures»,
передобробка, формування вікон, хронологічне розбиття, ознаки для базових моделей,
архітектура DS-1D-CNN і метрики.

TensorFlow імпортується лише всередині функцій, які його потребують, тож обробку даних
можна запускати й без нього.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------- Keras 2 (спільний з tfmot)
def _tf_version():
    from importlib.metadata import version, PackageNotFoundError
    for pkg in ("tensorflow", "tensorflow-cpu", "tensorflow-intel", "tensorflow-macos"):
        try:
            return tuple(int(x) for x in version(pkg).split(".")[:2])
        except (PackageNotFoundError, ValueError):
            continue
    return (0, 0)


def setup_keras_env():
    """Викликати ДО імпорту tfmot. Модель, QAT і конвертер мають працювати з ОДНИМ Keras 2.
    TF >= 2.16: tf.keras — це Keras 3, тому вмикаємо tf_keras (TF_USE_LEGACY_KERAS=1).
    TF 2.15: tf.keras уже Keras 2; ініціалізуємо його ДО імпорту tfmot, бо tfmot під час
    імпорту сам ставить TF_USE_LEGACY_KERAS=1 і інакше перемкнув би tf.keras на tf_keras
    (а tf_keras 2.15 при клонуванні моделі створює шари пакета keras — QAT падає)."""
    if _tf_version() >= (2, 16):
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
    else:
        os.environ.pop("TF_USE_LEGACY_KERAS", None)
        import tensorflow as tf
        _ = tf.keras.layers.Layer          # фіксуємо tf.keras = keras 2.15
        _ = tf.compat.v1.keras.layers.Layer
        _ = tf.compat.v2.keras.layers.Layer


def get_keras():
    from tensorflow_model_optimization.python.core.keras.compat import keras
    return keras


# ----------------------------------------------------------------------------- конфігурація
FS_RAW = 100          # Гц, частота дискретизації у наборі даних
DECIM = 10            # 100 Гц -> 10 Гц
WIN = 100             # відліків у вікні (10 с при 10 Гц)
HOP = 10              # крок вікна (1 с)
N_CH = 16             # МОН-сенсорів
N_CLS = 4             # 0 чисте повітря, 1 лише етилен, 2 лише CO/метан, 3 суміш
N_REG = 2             # [етилен, CO/метан], ppm
CLASS_NAMES = ["clean_air", "ethylene", "other_gas", "mixture"]

DATA_URL = ("https://archive.ics.uci.edu/static/public/322/"
            "gas+sensor+array+under+dynamic+gas+mixtures.zip")
RECORDINGS = {"co": "ethylene_CO.txt", "methane": "ethylene_methane.txt"}


@dataclass
class Splits:
    """Вікна після передобробки. raw_* — децимовані сирі сигнали (для HIL-тесту на МК)."""
    X: dict = field(default_factory=dict)        # split -> (M, WIN, N_CH) float32, нормовані
    raw: dict = field(default_factory=dict)      # split -> (M, WIN, N_CH) float32, сирі
    y_cls: dict = field(default_factory=dict)    # split -> (M,) int
    y_ppm: dict = field(default_factory=dict)    # split -> (M, 2) float32, ppm
    mean: np.ndarray | None = None               # (N_CH,) статистики навчальної вибірки
    std: np.ndarray | None = None
    conc_max: np.ndarray | None = None           # (2,) для нормування цілей регресії


# ----------------------------------------------------------------------------- дані
def download_dataset(data_dir: str) -> None:
    """Завантажує й розпаковує архів UCI, якщо файлів ще немає."""
    import urllib.request
    import zipfile

    os.makedirs(data_dir, exist_ok=True)
    if all(os.path.exists(os.path.join(data_dir, f)) for f in RECORDINGS.values()):
        return
    zpath = os.path.join(data_dir, "gas_dynamic_mixtures.zip")
    if not os.path.exists(zpath):
        print(f"Завантаження {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, zpath)
    with zipfile.ZipFile(zpath) as zf:
        for name in zf.namelist():
            base = os.path.basename(name)
            if base in RECORDINGS.values():
                with zf.open(name) as src, open(os.path.join(data_dir, base), "wb") as dst:
                    dst.write(src.read())
    # Деякі версії архіву містять вкладений zip
    if not all(os.path.exists(os.path.join(data_dir, f)) for f in RECORDINGS.values()):
        raise FileNotFoundError(
            "Не знайдено ethylene_CO.txt / ethylene_methane.txt. Розпакуйте архів вручну в "
            f"{data_dir} (сторінка набору: https://archive.ics.uci.edu/dataset/322)")


def load_recording(path: str, decim: int = DECIM, chunk_rows: int = 2_000_000):
    """
    Читає файл набору (стовпці: час, конц. CO/метану, конц. етилену, 16 сенсорів),
    децимує в decim разів і повертає:
      sensors (N, 16) — середнє за блоком (простий антиаліасинговий фільтр);
      ppm     (N, 2)  — [етилен, CO/метан] в останній момент блоку (уставки не розмиваються).
    Читання частинами, щоб не тримати 4 млн рядків у пам'яті у float64.
    """
    import pandas as pd

    sens_parts, ppm_parts = [], []
    carry = None
    reader = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None, dtype=np.float32,
                         chunksize=chunk_rows, engine="c")
    for chunk in reader:
        a = chunk.to_numpy(np.float32)
        if a.shape[1] != 19:
            raise ValueError(f"Очікувалось 19 стовпців, отримано {a.shape[1]} у {path}")
        if carry is not None:
            a = np.concatenate([carry, a])
        n = (len(a) // decim) * decim
        carry = a[n:]
        a = a[:n].reshape(-1, decim, 19)
        sens_parts.append(a[:, :, 3:].mean(axis=1))
        last = a[:, -1, :]
        ppm_parts.append(np.stack([last[:, 2], last[:, 1]], axis=1))  # [етилен, інший газ]
    return np.concatenate(sens_parts), np.concatenate(ppm_parts)


def transform(raw: np.ndarray) -> np.ndarray:
    """Нелінійне перетворення, яке повторює прошивка: log(1 + max(x, 0))."""
    return np.log1p(np.maximum(raw, 0.0)).astype(np.float32)


def normalize(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((transform(raw) - mean) / std).astype(np.float32)


def labels_from_ppm(ppm: np.ndarray, thr: float = 0.0) -> np.ndarray:
    eth = ppm[:, 0] > thr
    other = ppm[:, 1] > thr
    return (eth.astype(np.int64) + 2 * other.astype(np.int64))


def _windows(sig: np.ndarray, ppm: np.ndarray, win: int, hop: int):
    v = np.lib.stride_tricks.sliding_window_view(sig, (win, sig.shape[1]))[::hop, 0]
    ends = np.arange(win - 1, len(sig), hop)[: len(v)]
    return np.ascontiguousarray(v, dtype=np.float32), ppm[ends]


def make_splits(sensors: np.ndarray, ppm: np.ndarray, fractions=(0.70, 0.15, 0.15),
                win: int = WIN, hop: int = HOP, thr: float = 0.0) -> Splits:
    """
    Хронологічне розбиття 70/15/15 з буфером у одне вікно між частинами, щоб
    перекривні вікна не потрапили в різні вибірки (без витоку даних).
    Статистики нормування й максимум концентрацій беруться лише з навчальної частини.
    """
    n = len(sensors)
    b1 = int(n * fractions[0])
    b2 = int(n * (fractions[0] + fractions[1]))
    seg = {"train": (0, b1), "val": (b1 + win, b2), "test": (b2 + win, n)}

    s = Splits()
    tr = transform(sensors[seg["train"][0]:seg["train"][1]])
    s.mean = tr.mean(axis=0).astype(np.float32)
    s.std = (tr.std(axis=0) + 1e-6).astype(np.float32)
    s.conc_max = np.maximum(ppm[seg["train"][0]:seg["train"][1]].max(axis=0), 1e-6).astype(np.float32)

    for name, (a, b) in seg.items():
        raw_w, ppm_w = _windows(sensors[a:b], ppm[a:b], win, hop)
        s.raw[name] = raw_w
        s.X[name] = normalize(raw_w, s.mean, s.std)
        s.y_ppm[name] = ppm_w.astype(np.float32)
        s.y_cls[name] = labels_from_ppm(ppm_w, thr)
    return s


def synthetic_recording(hours: float = 2.0, seed: int = 0):
    """Штучний запис того ж формату (після децимації) — лише для перевірки конвеєра."""
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 * FS_RAW / DECIM)
    ppm = np.zeros((n, 2), np.float32)
    i = 0
    while i < n:
        L = rng.integers(400, 1200)
        ppm[i:i + L, 0] = rng.choice([0, rng.uniform(2, 20)])
        ppm[i:i + L, 1] = rng.choice([0, rng.uniform(25, 500)])
        i += L
    # перехідний процес сенсора (інерційна ланка) + різна чутливість каналів
    sens_gain = rng.uniform(0.2, 1.0, size=(2, N_CH)).astype(np.float32)
    drive = ppm / np.array([20.0, 500.0], np.float32) @ sens_gain
    y = np.zeros_like(drive)
    alpha = 0.05
    for k in range(1, n):
        y[k] = y[k - 1] + alpha * (drive[k] - y[k - 1])
    sensors = 2000.0 * (1.0 + 2.0 * y) + rng.normal(0, 20, size=y.shape)
    return sensors.astype(np.float32), ppm


# ----------------------------------------------------------------------------- ознаки для базових моделей
def window_features(X: np.ndarray) -> np.ndarray:
    """Середнє, стандартне відхилення, нахил (останній - перший) по кожному каналу -> 48 ознак."""
    return np.concatenate([X.mean(axis=1), X.std(axis=1), X[:, -1, :] - X[:, 0, :]], axis=1)


# ----------------------------------------------------------------------------- модель
def build_ds_cnn(win: int = WIN, n_ch: int = N_CH):
    """
    DS-1D-CNN з рис. 2 статті. Одновимірні згортки реалізовано як Conv2D/DepthwiseConv2D
    з ядром (k, 1) на вході (win, 1, n_ch): так модель підтримується QAT (tfmot)
    і в TFLM використовує оптимізовані ядра CMSIS-NN CONV_2D / DEPTHWISE_CONV_2D.
    Виходи: логіти класів (softmax рахується після деквантизації) і нормовані концентрації.
    """
    K = get_keras()
    layers, Input, Model = K.layers, K.Input, K.Model

    def cbr(x, layer):
        x = layer(x)
        x = layers.BatchNormalization(momentum=0.9)(x)   # швидше сходяться ковзні статистики
        return layers.ReLU()(x)

    inp = Input((win, 1, n_ch), name="window")
    x = cbr(inp, layers.Conv2D(16, (5, 1), padding="same", use_bias=False))
    x = cbr(x, layers.DepthwiseConv2D((5, 1), strides=(2, 1), padding="same", use_bias=False))
    x = cbr(x, layers.Conv2D(32, (1, 1), use_bias=False))
    x = cbr(x, layers.DepthwiseConv2D((3, 1), strides=(2, 1), padding="same", use_bias=False))
    x = cbr(x, layers.Conv2D(32, (1, 1), use_bias=False))
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation="relu")(x)
    cls = layers.Dense(N_CLS, name="cls")(x)
    conc = layers.Dense(N_REG, name="conc")(x)
    return Model(inp, [cls, conc], name="ds_1d_cnn")


def build_mlp(n_feat: int = 3 * N_CH):
    """Базовий MLP на 48 ознаках, кількість параметрів порівнянна з DS-1D-CNN (~5,4 тис.)."""
    K = get_keras()
    layers, Input, Model = K.layers, K.Input, K.Model

    inp = Input((n_feat,), name="features")
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dense(32, activation="relu")(x)
    return Model(inp, [layers.Dense(N_CLS, name="cls")(x), layers.Dense(N_REG, name="conc")(x)],
                 name="mlp")


def folded_param_count(win: int = WIN, n_ch: int = N_CH) -> tuple[int, int]:
    """Кількість параметрів після злиття BN і MAC на одне виведення (аналітично)."""
    t1, t2, t3 = win, -(-win // 2), -(-win // 4)
    p = (5 * n_ch * 16 + 16) + (5 * 16 + 16) + (16 * 32 + 32) + (3 * 32 + 32) + (32 * 32 + 32) \
        + (32 * 32 + 32) + (32 * N_CLS + N_CLS) + (32 * N_REG + N_REG)
    mac = t1 * 5 * n_ch * 16 + t2 * 5 * 16 + t2 * 16 * 32 + t3 * 3 * 32 + t3 * 32 * 32 \
        + 32 * 32 + 32 * (N_CLS + N_REG)
    return p, mac


# ----------------------------------------------------------------------------- метрики
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def metrics(y_cls, pred_cls, y_ppm, pred_ppm) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, r2_score

    out = {
        "accuracy_pct": 100.0 * accuracy_score(y_cls, pred_cls),
        "macro_f1": f1_score(y_cls, pred_cls, average="macro", labels=list(range(N_CLS)),
                             zero_division=0),
        "confusion": confusion_matrix(y_cls, pred_cls, labels=list(range(N_CLS))).tolist(),
    }
    for j, name in enumerate(["ethylene", "other_gas"]):
        out[f"mae_{name}_ppm"] = float(np.mean(np.abs(y_ppm[:, j] - pred_ppm[:, j])))
        out[f"r2_{name}"] = float(r2_score(y_ppm[:, j], pred_ppm[:, j]))
    return out
