"""
Хост для апаратної симуляції (hardware-in-the-loop): надсилає тестові вікна на SparkFun Edge,
приймає результати, порівнює їх з еталоном на ПК і рахує метрики для табл. 3 статті.

Приклади:
  python hil_host.py --port COM8 --n 500 --bench 1000              # плата (Windows)
  python hil_host.py --port /dev/ttyUSB0 --n 500 --bench 1000      # плата (Linux)
  python hil_host.py --exec ../firmware/host/gas_host --n 100       # перевірка без плати

Енергія (табл. 3): під час бенчмарку виміряйте профайлером середній струм I_act (мА)
при напрузі V (В), а в глибокому сні — I_sleep (мкА), і передайте:
  --v 3.3 --i-active-ma 1.0 --i-sleep-ua 6
Тоді E = V·I_act·t_inf, а середній струм для режиму «1 виведення/с»
I_avg = I_act·t_inf/1 c + I_sleep·(1 − t_inf/1 c).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SYNC = b"\xA5\x5A"


class SerialLink:
    def __init__(self, port, baud):
        import serial                                    # pip install pyserial
        self.s = serial.Serial(port, baud, timeout=10)
        time.sleep(0.5)
        self.s.reset_input_buffer()

    def send(self, b): self.s.write(b); self.s.flush()
    def readline(self): return self.s.readline().decode(errors="replace").strip()
    def close(self): self.s.close()


class ProcessLink:
    def __init__(self, exe):
        self.p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def send(self, b): self.p.stdin.write(b); self.p.stdin.flush()
    def readline(self): return self.p.stdout.readline().decode(errors="replace").strip()
    def close(self): self.p.stdin.close(); self.p.wait(timeout=5)


def read_until(link, prefix, max_lines=50):
    """Читає рядки, доки не трапиться потрібний префікс; рядки '#' — інформаційні."""
    info = {}
    for _ in range(max_lines):
        line = link.readline()
        if not line:
            raise TimeoutError(f"Немає відповіді (очікувався '{prefix}')")
        if line.startswith("E,"):
            raise RuntimeError(f"Помилка на пристрої: {line}")
        if line.startswith("#"):
            for kv in line[1:].split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    info[k] = v
        if line.startswith(prefix):
            return line, info
    raise TimeoutError(f"Не знайдено '{prefix}'")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--port", help="послідовний порт плати (COM8, /dev/ttyUSB0)")
    g.add_argument("--exec", help="хостова збірка прошивки (firmware/host/gas_host)")
    ap.add_argument("--baud", type=int, default=115200, help="має збігатися з HIL_BAUD у прошивці")
    ap.add_argument("--npz", default=os.path.join(HERE, "..", "results", "hil_test.npz"))
    ap.add_argument("--n", type=int, default=500, help="скільки тестових вікон надіслати")
    ap.add_argument("--bench", type=int, default=1000, help="кількість виведень у бенчмарку (0 — пропустити)")
    ap.add_argument("--v", type=float, help="напруга живлення, В")
    ap.add_argument("--i-active-ma", type=float, help="середній струм під час бенчмарку, мА")
    ap.add_argument("--i-sleep-ua", type=float, help="струм у глибокому сні, мкА")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "hil_results.json"))
    args = ap.parse_args()

    d = np.load(args.npz)
    raw = d["raw"].astype("<f4")
    n = min(args.n, len(raw))
    idx = np.linspace(0, len(raw) - 1, n).astype(int)        # рівномірно по тестовому відрізку

    link = SerialLink(args.port, args.baud) if args.port else ProcessLink(args.exec)
    try:
        link.send(SYNC + b"P")
        _, info = read_until(link, "# ready")
        print("Пристрій:", info)
        hz = float(info["cpu_hz"])

        rows = []
        t_start = time.time()
        for j, i in enumerate(idx):
            link.send(SYNC + b"I" + raw[i].tobytes())
            line, _ = read_until(link, "R,")
            v = [int(x) for x in line.split(",")[1:]]
            rows.append(v)
            if (j + 1) % 50 == 0:
                print(f"  {j + 1}/{n}  ({time.time() - t_start:.0f} c)")
        res = {"device": info, "n_windows": int(n)}
        if rows:
            R = np.array(rows)
            mcu_cls, mcu_raw_cls, mcu_raw_reg = R[:, 0], R[:, 1:5], R[:, 5:7]
            mcu_ppm = R[:, 7:9] / 1000.0
            cyc_pre, cyc_inf = R[:, 9].astype(float), R[:, 10].astype(float)

            res.update({"preprocess_ms_mean": float(cyc_pre.mean() / hz * 1e3),
                   "inference_ms_mean": float(cyc_inf.mean() / hz * 1e3),
                   "inference_ms_std": float(cyc_inf.std() / hz * 1e3),
                   "inference_cycles_mean": float(cyc_inf.mean())})

            if "y_cls" in d:
                y = d["y_cls"][idx]
                res["mcu_accuracy_pct"] = float(100 * (mcu_cls == y).mean())
                res["mcu_mae_ppm"] = np.abs(mcu_ppm - d["y_ppm"][idx]).mean(axis=0).tolist()
            if "pc_cls" in d:                                     # порівняння з еталоном TFLite на ПК
                res["class_agreement_with_pc_pct"] = float(100 * (mcu_cls == d["pc_cls"][idx]).mean())
                res["max_abs_diff_int8_logits"] = int(np.abs(mcu_raw_cls - d["pc_raw_cls"][idx]).max())
                res["max_abs_diff_int8_conc"] = int(np.abs(mcu_raw_reg - d["pc_raw_reg"][idx]).max())


        if args.bench > 0:
            link.send(SYNC + b"B" + struct.pack("<H", min(args.bench, 65535)))
            line, _ = read_until(link, "B,")
            _, bn, lo, hi, bhz = line.split(",")
            total = int(lo) + (int(hi) << 32)
            res["bench_n"] = int(bn)
            res["bench_inference_ms"] = total / int(bn) / int(bhz) * 1e3

        t_inf = res.get("bench_inference_ms", res.get("inference_ms_mean", 0.0)) / 1e3
        if args.v and args.i_active_ma:
            res["energy_uj_per_inference"] = args.v * args.i_active_ma * 1e-3 * t_inf * 1e6
            if args.i_sleep_ua is not None:
                duty = t_inf / 1.0
                res["avg_current_ua_1hz"] = args.i_active_ma * 1e3 * duty + args.i_sleep_ua * (1 - duty)
    finally:
        link.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("\nРядок для табл. 3 (Flash — з виводу arm-none-eabi-size, див. README):")
    print(f"| int8 | — | {int(info.get('arena_used', 0)) / 1024:.1f} | "
          f"{t_inf * 1e3:.2f} | "
          f"{res.get('energy_uj_per_inference', float('nan')):.1f} | "
          f"{res.get('avg_current_ua_1hz', float('nan')):.1f} |")


if __name__ == "__main__":
    sys.exit(main())
