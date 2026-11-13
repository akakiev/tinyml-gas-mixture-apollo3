#!/usr/bin/env bash
# Builds the three firmware images for the paper table (run inside WSL/Ubuntu).
#   SDK=~/AmbiqSuiteSDK TFLM=~/tflite-micro ./build_all.sh
# Output: builds/<row>.bin, builds/<row>.size (arm-none-eabi-size), builds/<row>.log, builds/versions.txt
set -euo pipefail
SDK=${SDK:-$HOME/AmbiqSuiteSDK}
TFLM=${TFLM:-$HOME/tflite-micro}
PROFILE_PIN=${PROFILE_PIN:-36}
TC=$TFLM/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin/arm-none-eabi
REPO=$(cd "$(dirname "$0")" && pwd)
OUT=$REPO/builds
mkdir -p "$OUT" "$REPO/tools"

[ -x "$TC-gcc" ] || { echo "ARM GCC not found at $TC-gcc (build TFLM first, step 4)"; exit 1; }
for k in cmsis_nn_gcc gcc; do
  L=$TFLM/gen/cortex_m_generic_cortex-m4+fp_release_$k/lib/libtensorflow-microlite.a
  [ -f "$L" ] || { echo "Missing TFLM library: $L (step 4)"; exit 1; }
done

# ASB uploader for flashing from Windows
rm -rf "$REPO/tools/asb"; cp -r "$SDK/boards_sfe/common/tools_sfe/asb" "$REPO/tools/asb"

build() {  # name variant kernels arena extra_defs
  local name=$1 variant=$2 kernels=$3 arena=$4 defs=$5
  echo "=== $name: variant=$variant kernels=$kernels arena=$arena defs=$defs"
  python3 "$REPO/python/use_variant.py" "$variant"
  ( cd "$REPO/firmware/gcc"
    make clean >/dev/null 2>&1 || true
    make SDKPATH="$SDK" TFLM_DIR="$TFLM" TOOLCHAIN="$TC" TFLM_KERNELS="$kernels" \
         ARENA_SIZE="$arena" EXTRA_DEFS="$defs" 2>&1 | tee "$OUT/$name.log"
    cp edge/bin/gas_tinyml_asb.bin "$OUT/$name.bin"
    "$TC-size" edge/bin/gas_tinyml_asb.axf | tee "$OUT/$name.size" )
}

build A_float32_48MHz float32  reference 65536 "-DPROFILE_PIN=$PROFILE_PIN"
build B_int8_48MHz    qat_int8 cmsis_nn  16384 "-DPROFILE_PIN=$PROFILE_PIN"
build C_int8_96MHz    qat_int8 cmsis_nn  16384 "-DUSE_BURST_MODE -DPROFILE_PIN=$PROFILE_PIN"
python3 "$REPO/python/use_variant.py" qat_int8

{ echo "tflite-micro: $(git -C "$TFLM" rev-parse HEAD)"
  echo "AmbiqSuiteSDK: $(git -C "$SDK" rev-parse HEAD)"
  echo "gcc: $("$TC-gcc" --version | sed -n 1p)"; } | tee "$OUT/versions.txt"
echo; echo "Flash = text + data:"; for f in "$OUT"/*.size; do echo "$(basename "$f" .size): $(tail -n1 "$f")"; done
