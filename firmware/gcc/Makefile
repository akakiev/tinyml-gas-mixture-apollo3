#******************************************************************************
# Прошивка gas_tinyml для SparkFun Edge (Ambiq Apollo3 Blue).
# На основі прикладу hello_world_uart_cpp з SparkFun AmbiqSuiteSDK.
#
#   make SDKPATH=~/AmbiqSuiteSDK TFLM_DIR=~/tflite-micro
#   make SDKPATH=... TFLM_DIR=... bootload_asb COM_PORT=COM8
#   make ... EXTRA_DEFS="-DUSE_BURST_MODE"             # TurboSPOT 96 МГц
#   make ... TFLM_KERNELS=reference ARENA_SIZE=65536    # float32 без CMSIS-NN
#   Між збірками різних варіантів завжди робіть make clean (зміна EXTRA_DEFS не перезбирає .o).
#   make ... EXTRA_DEFS="-DPROFILE_PIN=<вільний GPIO>"  # тригер для профайлера живлення
#
# Важливо: бібліотека TFLM і ця прошивка мають збиратися ОДНИМ компілятором
# (див. README, крок 3: TOOLCHAIN вказує на gcc, який завантажив TFLM).
#******************************************************************************

COM_PORT        ?=
ASB_UPLOAD_BAUD ?= 115200
SVL_UPLOAD_BAUD ?= 921600
PYTHON3         ?= python3
SDKPATH         ?=
BOARD           ?= edge
TFLM_DIR        ?=
# Яку бібліотеку TFLM брати: cmsis_nn (рядки int8) або reference (рядок float32 «без CMSIS-NN»).
# Обидві збірки лежать поруч у $(TFLM_DIR)/gen/, тому бібліотеку вибираємо явно, а не першу знайдену.
TFLM_KERNELS    ?= cmsis_nn
ifeq ($(TFLM_KERNELS),reference)
TFLM_LIB        ?= $(firstword $(wildcard $(TFLM_DIR)/gen/cortex_m_generic_cortex-m4+fp_release_gcc/lib/libtensorflow-microlite.a))
else
TFLM_LIB        ?= $(firstword $(wildcard $(TFLM_DIR)/gen/cortex_m_generic_cortex-m4+fp_release_$(TFLM_KERNELS)_gcc/lib/libtensorflow-microlite.a))
endif
$(info TFLM_LIB = $(TFLM_LIB))
# Мають збігатися з BUILD_TYPE бібліотеки TFLM (release)
TFLM_DEFS       ?= -DNDEBUG -DTF_LITE_STRIP_ERROR_STRINGS
ARENA_SIZE      ?= 16384
HIL_BAUD        ?= 115200
EXTRA_DEFS      ?=

ifneq ($(MAKECMDGOALS),clean)
ifeq ($(SDKPATH),)
    $(error Вкажіть SDKPATH=шлях до AmbiqSuiteSDK (git clone --recursive https://github.com/sparkfun/AmbiqSuiteSDK))
endif
ifeq ($(TFLM_DIR),)
    $(error Вкажіть TFLM_DIR=шлях до tflite-micro)
endif
ifeq ($(TFLM_LIB),)
    $(error Не знайдено libtensorflow-microlite.a для cortex-m4+fp і TFLM_KERNELS=$(TFLM_KERNELS) — див. HIL_KIT_README.md, крок 4)
endif
endif
ifeq ($(COM_PORT),)
    COM_PORT = COM4
endif

COMMONPATH  := $(SDKPATH)/boards_sfe/common
BOARDPATH   := $(SDKPATH)/boards_sfe/$(BOARD)
PROJECTPATH := ..
TARGET      := gas_tinyml
COMPILERNAME:= gcc
CONFIG      := $(PROJECTPATH)/gcc/$(BOARD)/bin

#### Defines / Includes / Sources ####
DEFINES  = -DPART_apollo3 -DAM_PACKAGE_BGA -DAM_PART_APOLLO3
DEFINES += -DTF_LITE_STATIC_MEMORY -DTFLM_CORTEX_M_GENERIC $(TFLM_DEFS)
DEFINES += -DARENA_SIZE=$(ARENA_SIZE) -DHIL_BAUD=$(HIL_BAUD) $(EXTRA_DEFS)

INCLUDES  = -I$(PROJECTPATH)/src
INCLUDES += -I$(BOARDPATH)/bsp
INCLUDES += -I$(SDKPATH) -I$(SDKPATH)/utils -I$(SDKPATH)/devices -I$(SDKPATH)/mcu/apollo3
INCLUDES += -I$(SDKPATH)/CMSIS/AmbiqMicro/Include -I$(SDKPATH)/CMSIS/ARM/Include
INCLUDES += -I$(TFLM_DIR)
INCLUDES += -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include
INCLUDES += -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/gemmlowp

SRC  = main.cc model_data.cc hal_apollo3.c
SRC += startup_gcc.c am_util_delay.c am_util_faultisr.c am_util_stdio.c

VPATH  = $(PROJECTPATH)/src $(SDKPATH)/utils $(COMMONPATH)/tools_sfe/templates

LIBS  = $(BOARDPATH)/bsp/gcc/bin/libam_bsp.a
LIBS += $(SDKPATH)/mcu/apollo3/hal/gcc/bin/libam_hal.a

ASB_UPLOADER = $(PYTHON3) $(COMMONPATH)/tools_sfe/asb/asb.py
SVL_UPLOADER = $(PYTHON3) $(COMMONPATH)/tools_sfe/svl/svl.py

#### Toolchain ####
SHELL     := /bin/bash
TOOLCHAIN ?= arm-none-eabi
CPU = cortex-m4
FPU = fpv4-sp-d16
FABI = hard
CC   = $(TOOLCHAIN)-gcc
CXX  = $(TOOLCHAIN)-g++
CP   = $(TOOLCHAIN)-objcopy
OD   = $(TOOLCHAIN)-objdump
SIZE = $(TOOLCHAIN)-size
RM   = rm

ARCH   = -mthumb -mcpu=$(CPU) -mfpu=$(FPU) -mfloat-abi=$(FABI)
COMMONFLAGS = $(ARCH) -ffunction-sections -fdata-sections -MMD -MP -Wall -g -O2 $(DEFINES) $(INCLUDES)
XFLAGS = $(COMMONFLAGS) -std=gnu++17 -fno-rtti -fno-exceptions -fno-threadsafe-statics -fno-use-cxa-atexit
CFLAGS = $(COMMONFLAGS) -std=c99
LFLAGS = $(ARCH) -nostartfiles -static --specs=nano.specs --specs=nosys.specs
LFLAGS += -Wl,--gc-sections,--entry,Reset_Handler,-Map,$(CONFIG)/$(TARGET).map
LFLAGS += -Wl,--start-group $(TFLM_LIB) -lstdc++ -lm -lc -lgcc $(LIBS) -Wl,--end-group

OBJS  = $(patsubst %.cc,$(CONFIG)/%.o,$(filter %.cc,$(SRC)))
OBJS += $(patsubst %.c,$(CONFIG)/%.o,$(filter %.c,$(SRC)))
DEPS  = $(OBJS:.o=.d)

#### Targets ####
all: asb
asb: directories $(CONFIG)/$(TARGET)_asb.bin size
svl: directories $(CONFIG)/$(TARGET)_svl.bin

directories:
	@mkdir -p $(CONFIG)

$(CONFIG)/%.o: %.cc
	@echo " Compiling $<"; $(CXX) -c $(XFLAGS) $< -o $@

$(CONFIG)/%.o: %.c
	@echo " Compiling $<"; $(CC) -c $(CFLAGS) $< -o $@

$(CONFIG)/$(TARGET)_asb.axf: LINKER_FILE = $(COMMONPATH)/tools_sfe/templates/asb_linker.ld
$(CONFIG)/$(TARGET)_asb.axf: $(OBJS) $(LIBS)
	@echo " Linking $@"; $(CXX) -Wl,-T,$(LINKER_FILE) -o $@ $(OBJS) $(LFLAGS)

$(CONFIG)/$(TARGET)_svl.axf: LINKER_FILE = $(COMMONPATH)/tools_sfe/templates/asb_svl_linker.ld
$(CONFIG)/$(TARGET)_svl.axf: $(OBJS) $(LIBS)
	@echo " Linking $@"; $(CXX) -Wl,-T,$(LINKER_FILE) -o $@ $(OBJS) $(LFLAGS)

$(CONFIG)/$(TARGET)_%.bin: $(CONFIG)/$(TARGET)_%.axf
	$(CP) -Obinary $< $@

# text+data = Flash, data+bss = статична RAM (без стеку) — для табл. 3
size: $(CONFIG)/$(TARGET)_asb.axf
	@$(SIZE) $<

bootload_asb: asb
	$(ASB_UPLOADER) --bin $(CONFIG)/$(TARGET)_asb.bin --load-address-blob 0x20000 --magic-num 0xCB \
	  -o $(CONFIG)/$(TARGET) --version 0x0 --load-address-wired 0xC000 -i 6 --options 0x1 \
	  -b $(ASB_UPLOAD_BAUD) -port $(COM_PORT) -r 2 -v

bootload_svl: svl
	$(SVL_UPLOADER) $(COM_PORT) -f $(CONFIG)/$(TARGET)_svl.bin -b $(SVL_UPLOAD_BAUD) -v

clean:
	$(RM) -rf $(CONFIG)

$(SDKPATH)/mcu/apollo3/hal/gcc/bin/libam_hal.a:
	$(MAKE) -C $(SDKPATH)/mcu/apollo3/hal/gcc TOOLCHAIN=$(TOOLCHAIN)

$(BOARDPATH)/bsp/gcc/bin/libam_bsp.a:
	$(MAKE) -C $(BOARDPATH)/bsp/gcc TOOLCHAIN=$(TOOLCHAIN)

-include $(DEPS)
.PHONY: all asb svl directories size bootload_asb bootload_svl clean
