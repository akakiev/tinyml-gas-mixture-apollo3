"""
Хост для апаратної симуляції (hardware-in-the-loop): надсилає тестові вікна на SparkFun Edge,
приймає результати, порівнює їх з еталоном на ПК і рахує метрики для табл. «Measurements on the
SparkFun Edge» статті.

Приклади (Windows, cmd):
  python hil_host.py --port COM8 --n 500 --bench 1000                 # точність + затримка
  python hil_host.py --port COM8 --n 0 --bench 30000                  # довгий бенчмарк -> I_act
  python hil_host.py --port COM8 --n 0 --bench 0 --duty 60            # 60 с «1 виведення/с» -> I_avg
  python hil_host.py --port COM8 --n 0 --bench 0 --sleep              # deep sleep -> I_sleep
  python hil_host.py --exec ../firmware/host/gas_host --n 100         # перевірка без плати

Після вимірювань профайлером передайте струми — скрипт порахує енергію:
  --v 3.3 --i-active-ma <I_act> --i-idle-ma <I_idle> --i-sleep-ua <I_sleep> --i-duty-ua <I_avg>
  E_gross = V·I_act·t_inf,  E_net = V·(I_act − I_idle)·t_inf,
  I_avg(модель) = I_act·t_inf/T + I_sleep·(1 − t_inf/T),  T = 1 с.
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
    eof_is_final = False

    def __init__(self, port, baud, boot_wait):
        import serial                                    # pip install pyserial
        self.s = serial.Serial(port, baud, timeout=2)
        # Відкриття порту смикає DTR -> RST плати: чекаємо, поки завантажувач передасть керування
        time.sleep(boot_wait)
        self.s.reset_input_buffer()

    def send(self, b): self.s.write(b); self.s.flush()
    def readline(self): return self.s.readline().decode(errors="replace")
    def close(self): self.s.close()


class ProcessLink:
    eof_is_final = True

    def __init__(self, exe):
        self.p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def send(self, b): self.p.stdin.write(b); self.p.stdin.flush()
    def readline(self): return self.p.stdout.readline().decode(errors="replace")
    def close(self): self.p.stdin.close(); self.p.wait(timeout=5)


def read_until(link, prefix, wait_s=15.0):
    """Читає рядки, доки не трапиться потрібний префікс; рядки '#' — інформаційні."""
    info = {}
    deadline = time.time() + wait_s
    while time.time() < deadline:
        raw = link.readline()
        if not raw:
            if link.eof_is_final:
                break
            continue                                      # тайм-аут читання — чекаємо далі
        line = "".join(ch for ch in raw if ch.isprintable()).strip()   # без сміття після сну
        if line.startswith("E,"):
            raise RuntimeError(f"Помилка на пристрої: {line}")
        if "#" in line:
            for kv in line[line.find("#") + 1:].split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    info[k] = v
        j = line.find(prefix)
        if j >= 0:
            return line[j:], info
    raise TimeoutError(f"Немає відповіді '{prefix}' за {wait_s:.0f} с")


def connect(link, tries=3):
    for t in range(tries):
        link.send(SYNC + b"P")
        try:
            return read_until(link, "# ready", wait_s=5)[1]
        except TimeoutError:
            print(f"  плата не відповіла (спроба {t + 1}/{tries}), повторюю...")
    raise TimeoutError("Плата не відповідає: перевірте порт, TX/RX, живлення і що прошивка запущена (RST без кнопки 14)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--port", help="послідовний порт плати (COM8, /dev/ttyUSB0)")
    g.add_argument("--exec", help="хостова збірка прошивки (firmware/host/gas_host)")
    ap.add_argument("--baud", type=int, default=115200, help="має збігатися з HIL_BAUD у прошивці")
    ap.add_argument("--boot-wait", type=float, default=1.5, help="пауза після відкриття порту, с")
    ap.add_argument("--npz", default=os.path.join(HERE, "..", "results", "hil_test.npz"))
    ap.add_argument("--n", type=int, default=500, help="скільки тестових вікон надіслати")
    ap.add_argument("--bench", type=int, default=1000, help="кількість виведень у бенчмарку (0 — пропустити)")
    ap.add_argument("--duty", type=int, default=0, help="кількість циклів режиму «сон -> виведення» (0 — пропустити)")
    ap.add_argument("--period-ms", type=int, default=1000, help="період режиму --duty, мс")
    ap.add_argument("--sleep", action="store_true", help="наприкінці перевести плату в deep sleep (I_sleep)")
    ap.add_argument("--v", type=float, help="напруга живлення, В")
    ap.add_argument("--i-active-ma", type=float, help="середній струм під час бенчмарку, мА")
    ap.add_argument("--i-idle-ma", type=float, help="струм в очікуванні команди (без виведення), мА")
    ap.add_argument("--i-sleep-ua", type=float, help="струм у глибокому сні, мкА")
    ap.add_argument("--i-duty-ua", type=float, help="виміряний середній струм у режимі --duty, мкА")
    ap.add_argument("--text", type=int, help="text з arm-none-eabi-size, Б")
    ap.add_argument("--data", type=int, help="data з arm-none-eabi-size, Б")
    ap.add_argument("--label", default="", help="підпис рядка таблиці (float32 / int8 48 MHz / ...)")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "hil_results.json"))
    args = ap.parse_args()

    d = np.load(args.npz)
    raw = d["raw"].astype("<f4")
    n = min(args.n, len(raw))
    idx = np.linspace(0, len(raw) - 1, n).astype(int)        # рівномірно по тестовому відрізку
    if n == 0 and (args.bench > 0 or args.duty > 0):
        idx = np.array([len(raw) // 2])                       # одне реальне вікно для бенчмарку

    link = SerialLink(args.port, args.baud, args.boot_wait) if args.port else ProcessLink(args.exec)
    res = {"label": args.label}
    try:
        info = connect(link)
        print("Пристрій:", info)
        hz = float(info["cpu_hz"])
        res.update({"device": info, "n_windows": int(n)})

        rows = []
        t_start = time.time()
        for j, i in enumerate(idx):
            link.send(SYNC + b"I" + raw[i].tobytes())
            line, _ = read_until(link, "R,")
            rows.append([int(x) for x in line.split(",")[1:12]])
            if (j + 1) % 50 == 0:
                print(f"  {j + 1}/{len(idx)}  ({time.time() - t_start:.0f} c)")
        if rows and n > 0:
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
                res["pc_accuracy_pct_same_windows"] = float(100 * (d["pc_cls"][idx] == d["y_cls"][idx]).mean())
                res["class_agreement_with_pc_pct"] = float(100 * (mcu_cls == d["pc_cls"][idx]).mean())
                res["max_abs_diff_int8_logits"] = int(np.abs(mcu_raw_cls - d["pc_raw_cls"][idx]).max())
                res["max_abs_diff_int8_conc"] = int(np.abs(mcu_raw_reg - d["pc_raw_reg"][idx]).max())

        if args.bench > 0:
            nb = min(args.bench, 65535)
            print(f"Бенчмарк: {nb} виведень... (саме зараз міряйте I_act)")
            link.send(SYNC + b"B" + struct.pack("<H", nb))
            line, _ = read_until(link, "B,", wait_s=30 + nb * 0.2)
            _, bn, lo, hi, bhz = line.split(",")[:5]
            total = int(lo) + (int(hi) << 32)
            res["bench_n"] = int(bn)
            res["bench_inference_ms"] = total / int(bn) / int(bhz) * 1e3

        if args.duty > 0:
            nd, T = min(args.duty, 65535), min(args.period_ms, 65535)
            print(f"Режим сон/виведення: {nd} циклів по {T} мс (~{nd * T / 1e3:.0f} с). "
                  f"Міряйте середній струм на профайлері за ціле число періодів.")
            link.send(SYNC + b"D" + struct.pack("<HH", nd, T))
            line, _ = read_until(link, "D,", wait_s=30 + nd * T / 1e3 * 1.2)
            p = line.split(",")
            total = int(p[3]) + (int(p[4]) << 32)
            res["duty_n"], res["duty_period_ms"] = int(p[1]), int(p[2])
            res["duty_work_ms"] = total / int(p[1]) / int(p[5]) * 1e3   # передобробка + виведення

        if args.sleep:
            link.send(SYNC + b"S")
            read_until(link, "# sleep", wait_s=5)
            print("Плата в deep sleep. Виміряйте I_sleep (усереднення ≥ 10 с), потім натисніть Enter.")
            print("Щоб вийти зі сну — кнопка RST (або перезапуск цього скрипта).")
            input()
    finally:
        link.close()

    # ------------------------------------------------------------------ енергія
    t_inf = res.get("bench_inference_ms", res.get("inference_ms_mean", 0.0)) / 1e3
    if args.v and args.i_active_ma:
        res["energy_uj_per_inference_gross"] = args.v * args.i_active_ma * 1e-3 * t_inf * 1e6
        if args.i_idle_ma is not None:
            res["energy_uj_per_inference_net"] = args.v * (args.i_active_ma - args.i_idle_ma) * 1e-3 * t_inf * 1e6
        if args.i_sleep_ua is not None:
            duty = t_inf / 1.0
            res["avg_current_ua_1hz_model"] = args.i_active_ma * 1e3 * duty + args.i_sleep_ua * (1 - duty)
    if args.i_duty_ua is not None:
        res["avg_current_ua_1hz_measured"] = args.i_duty_ua
        if args.v:
            T = res.get("duty_period_ms", args.period_ms) / 1e3
            res["energy_uj_per_period_measured"] = args.v * args.i_duty_ua * T
    if args.text is not None and args.data is not None:
        res["flash_bytes"] = args.text + args.data

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nЗбережено: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    sys.exit(main())
