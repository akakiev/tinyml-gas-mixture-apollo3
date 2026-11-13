"""
Повний конвеєр експерименту для статті:
  1) дані UCI -> децимація -> хронологічне розбиття 70/15/15 -> вікна 10 с;
  2) базові моделі: випадковий ліс і MLP на 48 ознаках;
  3) DS-1D-CNN float32 -> TFLite float32, int8 PTQ;
  4) QAT-донавчання -> TFLite int8;
  5) оцінка на тестовій вибірці (табл. 2 статті) -> results/results.json, results/table2.md;
  6) експорт обраної моделі в прошивку (firmware/src/model_data.cc, model_config.h)
     і тестових векторів для HIL (results/hil_test.npz).

Приклади:
  python train.py                       # запис етилен + CO (основний експеримент)
  python train.py --recording methane   # перевірка узагальнення (етилен + метан)
  python train.py --synthetic --epochs 3  # швидка перевірка середовища без завантаження даних
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

C.setup_keras_env()                     # до будь-якого імпорту TensorFlow

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIRMWARE_SRC = os.path.join(ROOT, "firmware", "src")
FIRMWARE_OPS = {"CONV_2D", "DEPTHWISE_CONV_2D", "MEAN", "FULLY_CONNECTED", "RESHAPE",
                "QUANTIZE", "DEQUANTIZE"}


# ----------------------------------------------------------------------------- аргументи
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recording", choices=list(C.RECORDINGS), default="co")
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    p.add_argument("--synthetic", action="store_true", help="штучні дані для перевірки конвеєра")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--qat-epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam", type=float, default=1.0, help="вага регресійної втрати λ")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--thr", type=float, default=0.0, help="поріг ppm для «газ присутній»")
    p.add_argument("--export", choices=["qat", "ptq", "float32", "auto"], default="auto",
                   help="яку int8-модель записати в прошивку (auto = кращий macro-F1 на val)")
    p.add_argument("--no-export", action="store_true",
                   help="не чіпати прошивку й HIL-файли (для запуску на записі з метаном)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ----------------------------------------------------------------------------- допоміжні
def set_seed(seed):
    import random
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def class_weights(y):
    cnt = np.bincount(y, minlength=C.N_CLS).astype(np.float64)
    w = np.where(cnt > 0, cnt.sum() / (C.N_CLS * np.maximum(cnt, 1)), 0.0)
    return w.astype(np.float32)


def as_input(X):
    return X[:, :, None, :]                 # (M, WIN, 1, N_CH)


def dataset(x, y_cls, y_reg, w, batch, shuffle):
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices(
        (x, (y_cls, y_reg), (w, np.ones(len(x), np.float32))))
    if shuffle:
        ds = ds.shuffle(min(len(x), 20000), reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(2)


def compile_model(model, lr, lam, steps):
    import tensorflow as tf
    K = C.get_keras()
    optimizers, losses = K.optimizers, K.losses
    sched = optimizers.schedules.CosineDecay(lr, decay_steps=max(steps, 1))
    model.compile(
        optimizer=optimizers.Adam(sched),
        loss=[losses.SparseCategoricalCrossentropy(from_logits=True), losses.Huber(delta=0.1)],
        loss_weights=[1.0, lam],
    )


def fit_xy(model, xtr, ctr, rtr, xva, cva, rva, args, epochs, lr):
    """Навчання з вагами класів (дисбаланс) і ранньою зупинкою за val_loss."""
    callbacks = C.get_keras().callbacks
    cw = class_weights(ctr)
    tr = dataset(xtr, ctr, rtr, cw[ctr], args.batch, True)
    va = dataset(xva, cva, rva, cw[cva], args.batch, False)
    steps = epochs * int(np.ceil(len(xtr) / args.batch))
    compile_model(model, lr, args.lam, steps)
    cb = [callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                  restore_best_weights=True)]
    return model.fit(tr, validation_data=va, epochs=epochs, callbacks=cb, verbose=2)


def fit(model, s, args, epochs, lr):
    return fit_xy(model, as_input(s.X["train"]), s.y_cls["train"], s.y_ppm["train"] / s.conc_max,
                  as_input(s.X["val"]), s.y_cls["val"], s.y_ppm["val"] / s.conc_max,
                  args, epochs, lr)


def predict_keras(model, X, conc_max):
    logits, conc = model.predict(as_input(X), batch_size=512, verbose=0)
    return logits.argmax(1), conc * conc_max


# ----------------------------------------------------------------------------- TFLite
def representative(X, n=500, seed=0):
    idx = np.random.default_rng(seed).choice(len(X), size=min(n, len(X)), replace=False)

    def gen():
        for i in idx:
            yield [as_input(X[i:i + 1]).astype(np.float32)]
    return gen


def convert(model, rep=None, int8=False):
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if int8:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()


class TFLiteRunner:
    """Виконання .tflite на ПК (еталон для порівняння з мікроконтролером)."""

    def __init__(self, model_bytes):
        import tensorflow as tf
        # Без XNNPACK: еталонні ядра TFLite ближчі до TFLM, а список операторів — без DELEGATE
        self.it = tf.lite.Interpreter(
            model_content=model_bytes,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        outs = self.it.get_output_details()
        self.o_cls = next(o for o in outs if o["shape"][-1] == C.N_CLS)
        self.o_reg = next(o for o in outs if o["shape"][-1] == C.N_REG)
        self.int8 = self.inp["dtype"] == np.int8

    def quantize_input(self, x):
        if not self.int8:
            return x.astype(np.float32)
        s, zp = self.inp["quantization"]
        return np.clip(np.round(x / s + zp), -128, 127).astype(np.int8)

    @staticmethod
    def _deq(o, v):
        if o["dtype"] == np.int8:
            s, zp = o["quantization"]
            return (v.astype(np.float32) - zp) * s
        return v

    def run(self, X):
        """X: (M, WIN, N_CH) нормовані. Повертає логіти, норм. концентрації і сирі int8-виходи."""
        logits, conc, raw_cls, raw_reg = [], [], [], []
        for i in range(len(X)):
            self.it.set_tensor(self.inp["index"], self.quantize_input(as_input(X[i:i + 1])))
            self.it.invoke()
            rc = self.it.get_tensor(self.o_cls["index"])[0]
            rr = self.it.get_tensor(self.o_reg["index"])[0]
            raw_cls.append(rc)
            raw_reg.append(rr)
            logits.append(self._deq(self.o_cls, rc))
            conc.append(self._deq(self.o_reg, rr))
        return np.array(logits), np.array(conc), np.array(raw_cls), np.array(raw_reg)

    def ops(self):
        try:
            return sorted({d["op_name"] for d in self.it._get_ops_details()})
        except Exception:                               # приватний API може змінитись
            return []


# ----------------------------------------------------------------------------- експорт у C
def write_c_array(data: bytes, path: str, name: str = "g_model"):
    lines = [", ".join(f"0x{b:02x}" for b in data[i:i + 12]) for i in range(0, len(data), 12)]
    with open(path, "w") as f:
        f.write("// Згенеровано python/train.py — не редагуйте вручну.\n")
        f.write('#include "model_data.h"\n\n')
        f.write(f"alignas(16) const unsigned char {name}[] = {{\n  " + ",\n  ".join(lines) + "\n};\n")
        f.write(f"const unsigned int {name}_len = {len(data)};\n")


def write_model_config(s: C.Splits, variant: str, path: str):
    arr = lambda a: ", ".join(f"{v:.8e}f" for v in a)
    with open(path, "w") as f:
        f.write(f"""// Згенеровано python/train.py — не редагуйте вручну.
// Варіант моделі: {variant}
#pragma once

#define GAS_WIN   {C.WIN}     // відліків у вікні (10 Гц)
#define GAS_N_CH  {C.N_CH}
#define GAS_N_CLS {C.N_CLS}
#define GAS_N_REG {C.N_REG}

// Передобробка: x_norm = (log(1 + max(x, 0)) - mean) / std  (статистики навчальної вибірки)
static const float kFeatMean[GAS_N_CH] = {{ {arr(s.mean)} }};
static const float kFeatStd[GAS_N_CH]  = {{ {arr(s.std)} }};

// Денормування регресійної голови: ppm = y * kConcMax  ([етилен, CO/метан])
static const float kConcMax[GAS_N_REG] = {{ {arr(s.conc_max)} }};
""")


def export_variant(blob, variant, mean, std, conc_max, tag, raw_test, y_cls, y_ppm, out_dir):
    """Записує модель у прошивку та еталонні виходи ПК для HIL-порівняння."""
    s = C.Splits(mean=mean, std=std, conc_max=conc_max)
    write_c_array(blob, os.path.join(FIRMWARE_SRC, "model_data.cc"))
    write_model_config(s, f"{variant} ({tag})", os.path.join(FIRMWARE_SRC, "model_config.h"))
    r = TFLiteRunner(blob)
    lg, cc, rc, rr = r.run(C.normalize(raw_test, mean, std))
    if not r.int8:                       # прошивка для float-моделі віддає значення ×1000
        rc, rr = np.round(rc * 1000).astype(np.int32), np.round(rr * 1000).astype(np.int32)
    np.savez_compressed(os.path.join(out_dir, "hil_test.npz"), raw=raw_test, y_cls=y_cls, y_ppm=y_ppm,
                        pc_raw_cls=rc, pc_raw_reg=rr, pc_cls=lg.argmax(1), pc_ppm=cc * conc_max)
    with open(os.path.join(out_dir, "model_firmware.tflite"), "wb") as f:
        f.write(blob)
    missing = set(r.ops()) - FIRMWARE_OPS
    if missing:
        print(f"УВАГА: оператори {missing} відсутні в резолвері main.cc — додайте їх.")
    print(f"Експортовано {variant} -> firmware/src/model_data.cc ({len(blob)} Б)")


# ----------------------------------------------------------------------------- головна
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(FIRMWARE_SRC, exist_ok=True)
    t0 = time.time()

    # 1. Дані
    if args.synthetic:
        sensors, ppm = C.synthetic_recording()
        tag = "synthetic"
    else:
        C.download_dataset(args.data_dir)
        cache = os.path.join(args.data_dir, f"{args.recording}_decim{C.DECIM}.npz")
        if os.path.exists(cache):
            d = np.load(cache)
            sensors, ppm = d["sensors"], d["ppm"]
        else:
            print("Читання й децимація запису (кілька хвилин)...")
            sensors, ppm = C.load_recording(os.path.join(args.data_dir, C.RECORDINGS[args.recording]))
            np.savez_compressed(cache, sensors=sensors, ppm=ppm)
        tag = args.recording
    s = C.make_splits(sensors, ppm, thr=args.thr)
    for k in ("train", "val", "test"):
        print(f"{k:5s}: {len(s.X[k]):6d} вікон, класи {np.bincount(s.y_cls[k], minlength=C.N_CLS).tolist()}")

    import tensorflow as tf
    set_seed(args.seed)
    res = {"recording": tag, "n_windows": {k: int(len(v)) for k, v in s.X.items()}}
    yt_cls, yt_ppm = s.y_cls["test"], s.y_ppm["test"]

    # 2. Базові моделі
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    F = {k: C.window_features(s.X[k]) for k in s.X}
    rf_c = RandomForestClassifier(200, n_jobs=None, class_weight="balanced", random_state=args.seed)
    rf_r = RandomForestRegressor(200, n_jobs=None, random_state=args.seed, min_samples_leaf=2)
    rf_c.fit(F["train"], s.y_cls["train"])
    rf_r.fit(F["train"], s.y_ppm["train"])
    res["random_forest"] = C.metrics(yt_cls, rf_c.predict(F["test"]), yt_ppm, rf_r.predict(F["test"]))
    res["random_forest"]["size_kb"] = (len(pickle.dumps(rf_c)) + len(pickle.dumps(rf_r))) / 1024

    sc = StandardScaler().fit(F["train"])
    mlp = C.build_mlp()
    fit_xy(mlp, sc.transform(F["train"]).astype(np.float32), s.y_cls["train"],
           s.y_ppm["train"] / s.conc_max, sc.transform(F["val"]).astype(np.float32),
           s.y_cls["val"], s.y_ppm["val"] / s.conc_max, args, args.epochs, args.lr)
    lg, cc = mlp.predict(sc.transform(F["test"]), verbose=0)
    res["mlp_float32"] = C.metrics(yt_cls, lg.argmax(1), yt_ppm, cc * s.conc_max)
    res["mlp_float32"]["size_kb"] = mlp.count_params() * 4 / 1024
    res["mlp_float32"]["params"] = int(mlp.count_params())

    # 3. DS-1D-CNN float32
    model = C.build_ds_cnn()
    model.summary(print_fn=lambda l: print(l))
    fit(model, s, args, args.epochs, args.lr)
    model.save(os.path.join(args.out_dir, "ds_cnn_float.h5"))
    pc, pp = predict_keras(model, s.X["test"], s.conc_max)
    res["ds_cnn_keras_float32"] = C.metrics(yt_cls, pc, yt_ppm, pp)
    p_fold, mac = C.folded_param_count()
    res["ds_cnn_analytic"] = {"params_after_bn_fold": p_fold, "macs_per_inference": mac}

    rep = representative(s.X["train"])
    variants = {"float32": convert(model), "ptq_int8": convert(model, rep, int8=True)}

    # 4. QAT
    import tensorflow_model_optimization as tfmot
    qat = tfmot.quantization.keras.quantize_model(model)
    fit(qat, s, args, args.qat_epochs, args.lr * 0.1)
    variants["qat_int8"] = convert(qat, rep, int8=True)

    # 5. Оцінка TFLite-моделей на тестовій і валідаційній вибірках
    val_f1 = {}
    runners = {}
    for name, blob in variants.items():
        with open(os.path.join(args.out_dir, f"ds_cnn_{name}.tflite"), "wb") as f:
            f.write(blob)
        r = TFLiteRunner(blob)
        runners[name] = r
        lg, cc, _, _ = r.run(s.X["test"])
        m = C.metrics(yt_cls, lg.argmax(1), yt_ppm, cc * s.conc_max)
        m["size_kb"] = len(blob) / 1024
        m["ops"] = r.ops()
        res[f"ds_cnn_tflite_{name}"] = m
        lgv, ccv, _, _ = r.run(s.X["val"])
        val_f1[name] = C.metrics(s.y_cls["val"], lgv.argmax(1), s.y_ppm["val"], ccv * s.conc_max)["macro_f1"]
        missing = set(m["ops"]) - FIRMWARE_OPS
        if missing:
            print(f"УВАГА: {name} містить оператори, яких немає в резолвері прошивки: {missing}")

    # 6. Експорт у прошивку + HIL-вектори
    chosen = {"qat": "qat_int8", "ptq": "ptq_int8", "float32": "float32"}.get(args.export) or \
        ("qat_int8" if val_f1["qat_int8"] >= val_f1["ptq_int8"] else "ptq_int8")
    if not args.no_export:
        np.savez(os.path.join(args.out_dir, "preproc.npz"), mean=s.mean, std=s.std, conc_max=s.conc_max,
                 tag=np.array(tag))
        export_variant(variants[chosen], chosen, s.mean, s.std, s.conc_max, tag,
                       s.raw["test"], yt_cls, yt_ppm, args.out_dir)
    res["exported_variant"] = chosen
    res["val_macro_f1"] = val_f1
    res["elapsed_s"] = time.time() - t0

    with open(os.path.join(args.out_dir, f"results_{tag}.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    write_table2(res, os.path.join(args.out_dir, f"table2_{tag}.md"))
    print(open(os.path.join(args.out_dir, f"table2_{tag}.md")).read())


def write_table2(res, path):
    rows = [("Випадковий ліс (ознаки)", "random_forest"), ("MLP float32", "mlp_float32"),
            ("DS-1D-CNN float32", "ds_cnn_tflite_float32"),
            ("DS-1D-CNN int8 (PTQ)", "ds_cnn_tflite_ptq_int8"),
            ("DS-1D-CNN int8 (QAT)", "ds_cnn_tflite_qat_int8")]
    out = ["| Модель | Accuracy, % | Macro-F1 | MAE етилен, ppm | MAE CO, ppm | R² етилен | R² CO | Розмір, КБ |",
           "|---|---|---|---|---|---|---|---|"]
    for label, k in rows:
        m = res[k]
        out.append(f"| {label} | {m['accuracy_pct']:.2f} | {m['macro_f1']:.3f} | "
                   f"{m['mae_ethylene_ppm']:.2f} | {m['mae_other_gas_ppm']:.1f} | "
                   f"{m['r2_ethylene']:.3f} | {m['r2_other_gas']:.3f} | {m['size_kb']:.1f} |")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
