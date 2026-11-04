// Самоперевірка без навченої моделі: будує int8 .tflite з топологією DS-1D-CNN (випадкові ваги)
// і записує test_src/model_data.cc + test_src/model_config.h. Справжні файли в ../src НЕ змінюються.
// Використання: make selftest
#include <cstdio>
#include <random>
#include <vector>
#include <string>
#include <cstring>
#include "tensorflow/lite/micro/schema/schema_generated.h"
using namespace tflite;

flatbuffers::FlatBufferBuilder fbb(1 << 20);
std::vector<flatbuffers::Offset<Buffer>> bufs;
std::vector<flatbuffers::Offset<Tensor>> tens;
std::mt19937 rng(1);

int buffer(const std::vector<uint8_t>& d) { bufs.push_back(CreateBufferDirect(fbb, d.empty() ? nullptr : &d)); return bufs.size() - 1; }
int tensor(std::vector<int> shape, TensorType t, std::vector<float> s, std::vector<int64_t> zp, int qdim, const std::vector<uint8_t>& data, const char* name) {
  int b = buffer(data);
  auto q = CreateQuantizationParametersDirect(fbb, nullptr, nullptr, s.empty() ? nullptr : &s, zp.empty() ? nullptr : &zp, QuantizationDetails_NONE, 0, qdim);
  tens.push_back(CreateTensorDirect(fbb, &shape, t, b, name, q));
  return tens.size() - 1;
}
std::vector<uint8_t> rnd_i8(int n) { std::vector<uint8_t> v(n); std::uniform_int_distribution<int> d(-60, 60); for (auto& x : v) x = (uint8_t)(int8_t)d(rng); return v; }
std::vector<uint8_t> zeros_i32(int n) { return std::vector<uint8_t>(4 * n, 0); }

const float A = 0.05f;  // activation scale
int act(std::vector<int> shape, bool relu, const char* n) { return tensor(shape, TensorType_INT8, {A}, {relu ? -128 : 0}, 0, {}, n); }
int weights(std::vector<int> shape, int ch, int qdim, const char* n) {
  int cnt = 1; for (int s : shape) cnt *= s;
  return tensor(shape, TensorType_INT8, std::vector<float>(ch, 0.01f), std::vector<int64_t>(ch, 0), qdim, rnd_i8(cnt), n);
}
int bias(int ch, const char* n) { return tensor({ch}, TensorType_INT32, std::vector<float>(ch, A * 0.01f), std::vector<int64_t>(ch, 0), 0, zeros_i32(ch), n); }

int main(int argc, char** argv) {
  buffer({});  // buffer 0 must be empty
  std::vector<flatbuffers::Offset<Operator>> ops;
  std::vector<flatbuffers::Offset<OperatorCode>> codes;
  BuiltinOperator opl[] = {BuiltinOperator_CONV_2D, BuiltinOperator_DEPTHWISE_CONV_2D, BuiltinOperator_MEAN, BuiltinOperator_FULLY_CONNECTED};
  for (auto o : opl) codes.push_back(CreateOperatorCode(fbb, (int8_t)o, 0, 1, o));

  int x = tensor({1, 100, 1, 16}, TensorType_INT8, {0.05f}, {0}, 0, {}, "window");
  int input = x;
  auto conv = [&](int in, int cin, int cout, int k, int stride, int tout, bool dw) {
    int w = dw ? weights({1, k, 1, cout}, cout, 3, "w") : weights({cout, k, 1, cin}, cout, 0, "w");
    int b = bias(cout, "b");
    int out = act({1, tout, 1, cout}, true, "a");
    std::vector<int> ins{in, w, b}, outs{out};
    if (dw) ops.push_back(CreateOperatorDirect(fbb, 1, &ins, &outs, BuiltinOptions_DepthwiseConv2DOptions,
        CreateDepthwiseConv2DOptions(fbb, Padding_SAME, 1, stride, 1, ActivationFunctionType_RELU).Union()));
    else ops.push_back(CreateOperatorDirect(fbb, 0, &ins, &outs, BuiltinOptions_Conv2DOptions,
        CreateConv2DOptions(fbb, k == 1 ? Padding_VALID : Padding_SAME, 1, stride, ActivationFunctionType_RELU).Union()));
    return out;
  };
  x = conv(x, 16, 16, 5, 1, 100, false);
  x = conv(x, 16, 16, 5, 2, 50, true);
  x = conv(x, 16, 32, 1, 1, 50, false);
  x = conv(x, 32, 32, 3, 2, 25, true);
  x = conv(x, 32, 32, 1, 1, 25, false);
  std::vector<uint8_t> axis(8); int32_t ax[2] = {1, 2}; memcpy(axis.data(), ax, 8);
  int axt = tensor({2}, TensorType_INT32, {}, {}, 0, axis, "axis");
  int m = act({1, 32}, true, "mean");
  { std::vector<int> ins{x, axt}, outs{m};
    ops.push_back(CreateOperatorDirect(fbb, 2, &ins, &outs, BuiltinOptions_ReducerOptions, CreateReducerOptions(fbb, false).Union())); }
  auto fc = [&](int in, int cin, int cout, bool relu, const char* n) {
    int cnt = cin * cout;
    int w = tensor({cout, cin}, TensorType_INT8, {0.01f}, {0}, 0, rnd_i8(cnt), "fw");
    int b = tensor({cout}, TensorType_INT32, {A * 0.01f}, {0}, 0, zeros_i32(cout), "fb");
    int out = relu ? act({1, cout}, true, n) : tensor({1, cout}, TensorType_INT8, {0.1f}, {0}, 0, {}, n);
    std::vector<int> ins{in, w, b}, outs{out};
    ops.push_back(CreateOperatorDirect(fbb, 3, &ins, &outs, BuiltinOptions_FullyConnectedOptions,
        CreateFullyConnectedOptions(fbb, relu ? ActivationFunctionType_RELU : ActivationFunctionType_NONE).Union()));
    return out;
  };
  int h = fc(m, 32, 32, true, "dense");
  int cls = fc(h, 32, 4, false, "cls");
  int reg = fc(h, 32, 2, false, "conc");
  std::vector<int> gin{input}, gout{cls, reg};
  auto sg = CreateSubGraphDirect(fbb, &tens, &gin, &gout, &ops, "main");
  std::vector<flatbuffers::Offset<SubGraph>> sgs{sg};
  auto model = CreateModelDirect(fbb, 3, &codes, &sgs, "test", &bufs);
  FinishModelBuffer(fbb, model);
  const uint8_t* p = fbb.GetBufferPointer();
  const unsigned n = fbb.GetSize();
  std::string dir = argc > 1 ? argv[1] : "test_src";
  FILE* f = fopen((dir + "/model_data.cc").c_str(), "w");
  if (!f) { perror("model_data.cc"); return 1; }
  fprintf(f, "#include \"model_data.h\"\nalignas(16) const unsigned char g_model[] = {");
  for (unsigned i = 0; i < n; ++i) fprintf(f, "%s0x%02x,", i % 16 ? "" : "\n  ", p[i]);
  fprintf(f, "\n};\nconst unsigned int g_model_len = %u;\n", n);
  fclose(f);
  f = fopen((dir + "/model_config.h").c_str(), "w");
  fprintf(f, "#pragma once\n#define GAS_WIN 100\n#define GAS_N_CH 16\n#define GAS_N_CLS 4\n#define GAS_N_REG 2\n"
             "static const float kFeatMean[16] = {8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8};\n"
             "static const float kFeatStd[16] = {.3f,.3f,.3f,.3f,.3f,.3f,.3f,.3f,.3f,.3f,.3f,.3f,.3f,.3f,.3f,.3f};\n"
             "static const float kConcMax[2] = {20.f, 500.f};\n");
  fclose(f);
  printf("test model: %u bytes\n", n);
}
