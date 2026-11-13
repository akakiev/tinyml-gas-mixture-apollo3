"""Extra analyses requested by Reviewer 1 on the real 'ethylene + CO' recording:
(a) size-constrained random forest, (b) int8 MLP on window features (+ op counts),
(c) chronological vs random window split for the DS-1D-CNN (leakage quantification).
Run from proj/:  python python/extra_real.py
"""
import os, sys, json, time, pickle, argparse
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
C.setup_keras_env()
import train as T

args = argparse.Namespace(batch=128, lr=1e-3, lam=1.0, patience=10, seed=42)
OUT = "results"; os.makedirs(OUT, exist_ok=True)
d = np.load("data/co_decim10.npz"); sensors, ppm = d["sensors"], d["ppm"]
s = C.make_splits(sensors, ppm)
yt, pt = s.y_cls["test"], s.y_ppm["test"]
res = {}
T.set_seed(args.seed)

# ---------------------------------------------------------------- (a) constrained random forest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
F = {k: C.window_features(s.X[k]) for k in s.X}
for tag, kw in [("rf_50x_d10", dict(n_estimators=50, max_depth=10)),
                ("rf_20x_d8", dict(n_estimators=20, max_depth=8))]:
    t0 = time.time()
    c = RandomForestClassifier(class_weight="balanced", random_state=args.seed, **kw).fit(F["train"], s.y_cls["train"])
    r = RandomForestRegressor(random_state=args.seed, **kw).fit(F["train"], s.y_ppm["train"])
    m = C.metrics(yt, c.predict(F["test"]), pt, r.predict(F["test"]))
    m["size_kb_pickle"] = (len(pickle.dumps(c)) + len(pickle.dumps(r))) / 1024
    m["nodes_total"] = int(sum(e.tree_.node_count for e in c.estimators_) + sum(e.tree_.node_count for e in r.estimators_))
    m["size_kb_flat_int16_est"] = m["nodes_total"] * 8 / 1024   # feature(1B)+threshold(2B)+children(2x2B)+value(1B)
    m["fit_s"] = time.time() - t0
    res[tag] = m; print(tag, {k: m[k] for k in ("accuracy_pct", "macro_f1", "mae_ethylene_ppm", "mae_other_gas_ppm", "size_kb_pickle")}, flush=True)

# ---------------------------------------------------------------- (b) MLP float32 + int8
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(F["train"])
Fs = {k: sc.transform(v).astype(np.float32) for k, v in F.items()}
mlp = C.build_mlp()
T.fit_xy(mlp, Fs["train"], s.y_cls["train"], s.y_ppm["train"] / s.conc_max,
         Fs["val"], s.y_cls["val"], s.y_ppm["val"] / s.conc_max, args, 60, args.lr)
lg, cc = mlp.predict(Fs["test"], verbose=0)
res["mlp_float32"] = C.metrics(yt, lg.argmax(1), pt, cc * s.conc_max)
res["mlp_float32"]["params"] = int(mlp.count_params())
import tensorflow as tf
def rep():
    idx = np.random.default_rng(0).choice(len(Fs["train"]), 500, replace=False)
    for i in idx:
        yield [Fs["train"][i:i + 1]]
conv = tf.lite.TFLiteConverter.from_keras_model(mlp)
conv.optimizations = [tf.lite.Optimize.DEFAULT]; conv.representative_dataset = rep
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8; conv.inference_output_type = tf.int8
blob = conv.convert(); open(os.path.join(OUT, "mlp_int8.tflite"), "wb").write(blob)
it = tf.lite.Interpreter(model_content=blob,
                         experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
it.allocate_tensors(); inp = it.get_input_details()[0]; outs = it.get_output_details()
oc = next(o for o in outs if o["shape"][-1] == C.N_CLS); orr = next(o for o in outs if o["shape"][-1] == C.N_REG)
si, zi = inp["quantization"]; L, R = [], []
for i in range(len(Fs["test"])):
    it.set_tensor(inp["index"], np.clip(np.round(Fs["test"][i:i + 1] / si + zi), -128, 127).astype(np.int8)); it.invoke()
    a = it.get_tensor(oc["index"])[0].astype(np.float32); b = it.get_tensor(orr["index"])[0].astype(np.float32)
    L.append((a - oc["quantization"][1]) * oc["quantization"][0]); R.append((b - orr["quantization"][1]) * orr["quantization"][0])
res["mlp_int8"] = C.metrics(yt, np.array(L).argmax(1), pt, np.array(R) * s.conc_max)
res["mlp_int8"]["size_kb"] = len(blob) / 1024
win, nch = C.WIN, C.N_CH
res["op_counts"] = {
    "features_ops_per_window_approx": int(nch * (win + 2 * win + 1)),   # mean (100 add), std (100 mul+100 add), slope (1 sub)
    "mlp_macs": int(48 * 64 + 64 * 32 + 32 * C.N_CLS + 32 * C.N_REG),
    "ds_cnn_macs": int(C.folded_param_count()[1]),
}
print("mlp", {k: res[k]["accuracy_pct"] for k in ("mlp_float32", "mlp_int8")}, res["op_counts"], flush=True)

# ---------------------------------------------------------------- (c) chronological vs random split (CNN float32)
EP = 30
def train_eval(split):
    T.set_seed(args.seed)
    m = C.build_ds_cnn(); args.patience = 5
    T.fit(m, split, args, EP, args.lr)
    pc, pp = T.predict_keras(m, split.X["test"], split.conc_max)
    return C.metrics(split.y_cls["test"], pc, split.y_ppm["test"], pp)

res["split_chronological"] = train_eval(s)
print("chrono", res["split_chronological"]["accuracy_pct"], flush=True)
raw, pw = C._windows(sensors, ppm, C.WIN, C.HOP)
ycls = C.labels_from_ppm(pw)
perm = np.random.default_rng(args.seed).permutation(len(raw))
n1, n2 = int(0.70 * len(raw)), int(0.85 * len(raw))
idx = {"train": perm[:n1], "val": perm[n1:n2], "test": perm[n2:]}
r_ = C.Splits()
tr = C.transform(raw[idx["train"]]).reshape(-1, C.N_CH)
r_.mean = tr.mean(0).astype(np.float32); r_.std = (tr.std(0) + 1e-6).astype(np.float32); del tr
r_.conc_max = np.maximum(pw[idx["train"]].max(0), 1e-6).astype(np.float32)
for k, ii in idx.items():
    r_.X[k] = C.normalize(raw[ii], r_.mean, r_.std); r_.y_cls[k] = ycls[ii]; r_.y_ppm[k] = pw[ii].astype(np.float32)
del raw
res["split_random"] = train_eval(r_)
res["split_random"]["n_windows"] = {k: int(len(v)) for k, v in idx.items()}
res["split_epochs"] = EP
print("random", res["split_random"]["accuracy_pct"], flush=True)
json.dump(res, open(os.path.join(OUT, "extra_real_co.json"), "w"), indent=1)
print("done")
