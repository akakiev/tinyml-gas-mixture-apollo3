# Збірка прошивки для ПК (Linux/macOS/WSL) — перевірка без плати.
#   1) зібрати TFLM для хоста:
#        cd $(TFLM_DIR) && make -f tensorflow/lite/micro/tools/make/Makefile BUILD_TYPE=release microlite
#   2) make TFLM_DIR=/шлях/до/tflite-micro
#   3) python ../../python/hil_host.py --exec ./gas_host --npz ../../results/hil_test.npz
TFLM_DIR ?= ../../../tflite-micro
TFLM_LIB ?= $(firstword $(wildcard $(TFLM_DIR)/gen/*_x86_64_release*/lib/libtensorflow-microlite.a) \
                        $(wildcard $(TFLM_DIR)/gen/*/lib/libtensorflow-microlite.a))
SRC_DIR  := ../src
# Мають збігатися з BUILD_TYPE бібліотеки: release -> ці два прапорці, default -> порожньо
TFLM_DEFS ?= -DNDEBUG -DTF_LITE_STRIP_ERROR_STRINGS

CXXFLAGS := -std=c++17 -O2 -Wall -fno-rtti -fno-exceptions -DTF_LITE_STATIC_MEMORY $(TFLM_DEFS) \
            -I$(SRC_DIR) -I$(TFLM_DIR) \
            -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
            -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/gemmlowp
CFLAGS   := -std=c99 -O2 -Wall -I$(SRC_DIR)

gas_host: main.o model_data.o hal_host.o
	$(CXX) -o $@ $^ $(TFLM_LIB) -lm

main.o: $(SRC_DIR)/main.cc $(SRC_DIR)/model_config.h $(SRC_DIR)/hal.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
model_data.o: $(SRC_DIR)/model_data.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@
hal_host.o: hal_host.c $(SRC_DIR)/hal.h
	$(CC) $(CFLAGS) -c $< -o $@

# ---- самоперевірка без навченої моделі (випадкові ваги, та сама топологія)
FB_INC := -I$(TFLM_DIR) -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include
gen_test_model: gen_test_model.cc
	$(CXX) -std=c++17 -O1 $(FB_INC) $< -o $@

test_src/model_data.cc: gen_test_model
	mkdir -p test_src && ./gen_test_model test_src

gas_host_test: $(SRC_DIR)/main.cc test_src/model_data.cc hal_host.o
	$(CXX) $(CXXFLAGS:-I$(SRC_DIR)=-Itest_src -I$(SRC_DIR)) $(SRC_DIR)/main.cc test_src/model_data.cc hal_host.o -o $@ $(TFLM_LIB) -lm

selftest: gas_host_test
	printf '\245\132P\245\132B\350\003' | ./gas_host_test

clean:
	rm -rf *.o gas_host gas_host_test gen_test_model test_src
.PHONY: clean selftest
