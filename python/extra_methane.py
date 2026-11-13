
import os, sys, json, pickle, argparse
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
C.setup_keras_env()
import train as T
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
args = argparse.Namespace(batch=128, lr=1e-3, lam=1.0, patience=10, seed=42)
d=np.load("data/methane_decim10.npz"); s=C.make_splits(d["sensors"], d["ppm"])
yt,pt=s.y_cls["test"],s.y_ppm["test"]; T.set_seed(42); res={}
F={k:C.window_features(s.X[k]) for k in s.X}
c=RandomForestClassifier(n_estimators=20,max_depth=8,class_weight="balanced",random_state=42).fit(F["train"],s.y_cls["train"])
r=RandomForestRegressor(n_estimators=20,max_depth=8,random_state=42).fit(F["train"],s.y_ppm["train"])
m=C.metrics(yt,c.predict(F["test"]),pt,r.predict(F["test"]))
m["nodes_total"]=int(sum(e.tree_.node_count for e in c.estimators_)+sum(e.tree_.node_count for e in r.estimators_)); m["size_kb_flat_int16_est"]=m["nodes_total"]*8/1024
res["rf_20x_d8"]=m
sc=StandardScaler().fit(F["train"]); Fs={k:sc.transform(v).astype(np.float32) for k,v in F.items()}
mlp=C.build_mlp()
T.fit_xy(mlp,Fs["train"],s.y_cls["train"],s.y_ppm["train"]/s.conc_max,Fs["val"],s.y_cls["val"],s.y_ppm["val"]/s.conc_max,args,60,args.lr)
def rep():
    idx=np.random.default_rng(0).choice(len(Fs["train"]),500,replace=False)
    for i in idx: yield [Fs["train"][i:i+1]]
conv=tf.lite.TFLiteConverter.from_keras_model(mlp); conv.optimizations=[tf.lite.Optimize.DEFAULT]; conv.representative_dataset=rep
conv.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]; conv.inference_input_type=tf.int8; conv.inference_output_type=tf.int8
blob=conv.convert()
it=tf.lite.Interpreter(model_content=blob,experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES); it.allocate_tensors()
inp=it.get_input_details()[0]; outs=it.get_output_details()
oc=next(o for o in outs if o["shape"][-1]==C.N_CLS); orr=next(o for o in outs if o["shape"][-1]==C.N_REG)
si,zi=inp["quantization"]; L,R=[],[]
for i in range(len(Fs["test"])):
    it.set_tensor(inp["index"],np.clip(np.round(Fs["test"][i:i+1]/si+zi),-128,127).astype(np.int8)); it.invoke()
    L.append((it.get_tensor(oc["index"])[0].astype(np.float32)-oc["quantization"][1])*oc["quantization"][0])
    R.append((it.get_tensor(orr["index"])[0].astype(np.float32)-orr["quantization"][1])*orr["quantization"][0])
res["mlp_int8"]=C.metrics(yt,np.array(L).argmax(1),pt,np.array(R)*s.conc_max); res["mlp_int8"]["size_kb"]=len(blob)/1024
json.dump(res,open("results/extra_real_methane.json","w"),indent=1); print("done")
