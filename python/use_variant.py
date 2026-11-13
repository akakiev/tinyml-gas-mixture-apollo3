"""Перемикає модель у прошивці на один із готових варіантів (variants/<name>/). Лише stdlib.

  python3 python/use_variant.py qat_int8     # модель статті (за замовчуванням)
  python3 python/use_variant.py float32      # рядок вартості float32
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
name = sys.argv[1] if len(sys.argv) > 1 else "qat_int8"
src = os.path.join(ROOT, "variants", name)
if not os.path.isdir(src):
    sys.exit(f"Немає варіанта {name}; є: {sorted(os.listdir(os.path.join(ROOT, 'variants')))}")
for f, dst in (("model_data.cc", "firmware/src"), ("model_config.h", "firmware/src"), ("hil_test.npz", "results")):
    shutil.copyfile(os.path.join(src, f), os.path.join(ROOT, dst, f))   # copyfile оновлює mtime -> make перезбирає
print(f"Варіант {name}: firmware/src/model_data.cc, model_config.h, results/hil_test.npz оновлено")
