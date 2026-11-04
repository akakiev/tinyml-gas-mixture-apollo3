"""
Перемикає модель у прошивці без повторного навчання (потрібно для різних рядків табл. 3).
Використовує файли, які створив train.py: results/ds_cnn_<variant>.tflite, preproc.npz, hil_test.npz.

  python export_firmware.py --variant float32     # рядок «float32 (TFLM, без CMSIS-NN)»
  python export_firmware.py --variant qat_int8    # повернути int8
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--variant", choices=["float32", "ptq_int8", "qat_int8"], required=True)
ap.add_argument("--out-dir", default=os.path.join(T.ROOT, "results"))
a = ap.parse_args()

pre = np.load(os.path.join(a.out_dir, "preproc.npz"))
hil = np.load(os.path.join(a.out_dir, "hil_test.npz"))
blob = open(os.path.join(a.out_dir, f"ds_cnn_{a.variant}.tflite"), "rb").read()
T.export_variant(blob, a.variant, pre["mean"], pre["std"], pre["conc_max"], str(pre["tag"]),
                 hil["raw"], hil["y_cls"], hil["y_ppm"], a.out_dir)
