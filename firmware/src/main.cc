// TinyML-розпізнавання складу газової суміші на Ambiq Apollo3 Blue (SparkFun Edge).
//
// Режим апаратної симуляції (HIL): ПК надсилає через UART вікна децимованих сирих
// сигналів 16 сенсорів, мікроконтролер виконує ту саму передобробку, що й у статті
// (log(1+x), z-нормування, квантизація int8), запускає DS-1D-CNN у TFLM і повертає
// клас, концентрації та кількість тактів.
//
// Протокол (усі числа little-endian):
//   ПК -> МК:  A5 5A 'P'                         — інформація про модель
//              A5 5A 'I' + float32[WIN*N_CH]     — виведення на одному вікні
//              A5 5A 'B' + uint16 n              — бенчмарк: n виведень на останньому вікні
//   МК -> ПК:  текстові рядки:
//              "# key=value ..."                 — інформація
//              "R,cls,l0,l1,l2,l3,r0,r1,eth_mppm,gas_mppm,cyc_pre,cyc_inf"
//                 l* і r* — сирі int8-виходи (для побітового порівняння з ПК),
//                 *_mppm — концентрації в мілі-ppm, cyc_* — такти передобробки й виведення
//              "B,n,total_cycles_lo,total_cycles_hi,cpu_hz"
//              "E,<код>"                        — помилка
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"

#include "hal.h"
#include "model_config.h"
#include "model_data.h"

#ifdef TFLM_CORTEX_M_GENERIC
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
#endif

#ifndef ARENA_SIZE
#define ARENA_SIZE (16 * 1024)
#endif

namespace {

constexpr uint8_t kSync0 = 0xA5, kSync1 = 0x5A;
constexpr int kWinLen = GAS_WIN * GAS_N_CH;

alignas(16) uint8_t g_arena[ARENA_SIZE];
float g_raw[kWinLen];
tflite::MicroInterpreter *g_interp = nullptr;
TfLiteTensor *g_in = nullptr, *g_cls = nullptr, *g_reg = nullptr;

// ---------------------------------------------------------------- вивід без printf
struct Line {
  char buf[160];
  int n = 0;
  Line &s(const char *t) { while (*t && n < (int)sizeof(buf) - 2) buf[n++] = *t++; return *this; }
  Line &u(uint32_t v) {
    char tmp[11]; int k = 0;
    do { tmp[k++] = char('0' + v % 10); v /= 10; } while (v);
    while (k && n < (int)sizeof(buf) - 2) buf[n++] = tmp[--k];
    return *this;
  }
  Line &i(int32_t v) { if (v < 0) { s("-"); return u(uint32_t(-(int64_t)v)); } return u(uint32_t(v)); }
  Line &c() { return s(","); }
  void send() { buf[n++] = '\n'; hal_write(buf, n); n = 0; }
};

void debug_log(const char *s) { hal_write(s, strlen(s)); }

// ---------------------------------------------------------------- модель
TfLiteTensor *find_output(int last_dim) {
  for (size_t k = 0; k < g_interp->outputs_size(); ++k) {
    TfLiteTensor *t = g_interp->output(k);
    if (t->dims->data[t->dims->size - 1] == last_dim) return t;
  }
  return nullptr;
}

int setup_model() {
  const tflite::Model *model = tflite::GetModel(g_model);
  if (g_model_len == 0) return 1;
  if (model->version() != TFLITE_SCHEMA_VERSION) return 2;

  // Оператори, які генерує конвертер для DS-1D-CNN (див. FIRMWARE_OPS у train.py)
  static tflite::MicroMutableOpResolver<7> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();

  static tflite::MicroInterpreter interp(model, resolver, g_arena, ARENA_SIZE);
  g_interp = &interp;
  if (interp.AllocateTensors() != kTfLiteOk) return 3;

  g_in = interp.input(0);
  g_cls = find_output(GAS_N_CLS);
  g_reg = find_output(GAS_N_REG);
  if (!g_cls || !g_reg) return 4;
  int in_elems = 1;
  for (int d = 0; d < g_in->dims->size; ++d) in_elems *= g_in->dims->data[d];
  if (in_elems != kWinLen) return 5;
  if (g_in->type != kTfLiteInt8 && g_in->type != kTfLiteFloat32) return 6;
  return 0;
}

// Передобробка, ідентична common.normalize() + TFLiteRunner.quantize_input() у Python
void preprocess(const float *raw) {
  if (g_in->type == kTfLiteInt8) {
    const float inv_s = 1.0f / g_in->params.scale;
    const int zp = g_in->params.zero_point;
    int8_t *dst = g_in->data.int8;
    for (int i = 0; i < kWinLen; ++i) {
      const int c = i % GAS_N_CH;
      const float x = raw[i] > 0.0f ? raw[i] : 0.0f;
      const float v = (log1pf(x) - kFeatMean[c]) / kFeatStd[c];
      long q = lroundf(v * inv_s) + zp;
      dst[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }
  } else {
    float *dst = g_in->data.f;
    for (int i = 0; i < kWinLen; ++i) {
      const int c = i % GAS_N_CH;
      const float x = raw[i] > 0.0f ? raw[i] : 0.0f;
      dst[i] = (log1pf(x) - kFeatMean[c]) / kFeatStd[c];
    }
  }
}

float dequant(const TfLiteTensor *t, int k) {
  if (t->type == kTfLiteInt8) return (t->data.int8[k] - t->params.zero_point) * t->params.scale;
  return t->data.f[k];
}
int32_t raw_out(const TfLiteTensor *t, int k) {
  return t->type == kTfLiteInt8 ? t->data.int8[k] : (int32_t)lroundf(t->data.f[k] * 1000.0f);
}

void cmd_info() {
  Line l;
  l.s("# model_bytes=").u(g_model_len).s(" arena_used=").u(g_interp->arena_used_bytes())
   .s(" arena_size=").u(ARENA_SIZE).s(" cpu_hz=").u(hal_cpu_hz()).send();
  l.s("# in_type=").s(g_in->type == kTfLiteInt8 ? "int8" : "float32")
   .s(" win=").u(GAS_WIN).s(" ch=").u(GAS_N_CH).send();
  l.s("# ready").send();
}

void cmd_infer() {
  if (hal_read(reinterpret_cast<uint8_t *>(g_raw), sizeof(g_raw)) != 0) { Line().s("E,read").send(); return; }

  const uint32_t t0 = hal_cycles();
  preprocess(g_raw);
  const uint32_t t1 = hal_cycles();
  hal_profile_pin(1);
  const TfLiteStatus st = g_interp->Invoke();
  hal_profile_pin(0);
  const uint32_t t2 = hal_cycles();
  if (st != kTfLiteOk) { Line().s("E,invoke").send(); return; }

  int best = 0;
  float best_v = dequant(g_cls, 0);
  for (int k = 1; k < GAS_N_CLS; ++k) {
    const float v = dequant(g_cls, k);
    if (v > best_v) { best_v = v; best = k; }
  }
  Line l;
  l.s("R,").i(best);
  for (int k = 0; k < GAS_N_CLS; ++k) l.c().i(raw_out(g_cls, k));
  for (int k = 0; k < GAS_N_REG; ++k) l.c().i(raw_out(g_reg, k));
  for (int k = 0; k < GAS_N_REG; ++k) {
    float ppm = dequant(g_reg, k) * kConcMax[k];
    if (ppm < 0.0f) ppm = 0.0f;                         // фізично неможливі від'ємні значення
    l.c().i((int32_t)lroundf(ppm * 1000.0f));
  }
  l.c().u(t1 - t0).c().u(t2 - t1).send();
}

void cmd_bench() {
  uint8_t b[2];
  if (hal_read(b, 2) != 0) { Line().s("E,read").send(); return; }
  const uint32_t n = b[0] | (uint32_t(b[1]) << 8);
  preprocess(g_raw);
  uint64_t total = 0;
  hal_profile_pin(1);
  for (uint32_t k = 0; k < n; ++k) {
    const uint32_t t0 = hal_cycles();
    g_interp->Invoke();
    total += uint32_t(hal_cycles() - t0);           // різниця коректна й при переповненні CYCCNT
  }
  hal_profile_pin(0);
  Line().s("B,").u(n).c().u(uint32_t(total)).c().u(uint32_t(total >> 32)).c().u(hal_cpu_hz()).send();
}

}  // namespace

int main() {
  hal_init();
  tflite::InitializeTarget();
#ifdef TFLM_CORTEX_M_GENERIC
  RegisterDebugLogCallback(debug_log);
#else
  (void)debug_log;
#endif

  const int err = setup_model();
  if (err) {
    while (true) { Line().s("E,setup,").i(err).send(); for (volatile uint32_t d = 0; d < 20000000; ++d) {} }
  }
  memset(g_raw, 0, sizeof(g_raw));
  cmd_info();

  // Головний цикл: пошук синхропослідовності A5 5A, далі команда
  uint8_t prev = 0, cur = 0;
  while (true) {
    if (hal_read(&cur, 1) != 0) return 0;               // на ПК: кінець вхідного потоку
    if (prev == kSync0 && cur == kSync1) {
      uint8_t cmd;
      if (hal_read(&cmd, 1) != 0) return 0;
      if (cmd == 'I') cmd_infer();
      else if (cmd == 'B') cmd_bench();
      else if (cmd == 'P') cmd_info();
      else Line().s("E,cmd").send();
      cur = 0;
    }
    prev = cur;
  }
}
