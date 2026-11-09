# Redukt GPU CUDA Plugin — build libredukt_gpu.so (Linux, CUDA Toolkit 12.x)
#
# Usage: make            -> libredukt_gpu.so (sm_75 T4 + sm_89 L4 SASS, compute_89 PTX fallback)
#        make clean
#        make info
#   or:  ./build.sh      -> same, inside nvidia/cuda:12.4.1-devel-ubuntu22.04 when nvcc is absent
#
# Flags (NVIDIA nvcc docs, https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html):
#   --generate-code arch=compute_XX,code=[sm_XX,compute_XX]  #using-code-generation-options
#       (one entry per virtual architecture; may be repeated, #generate-code-specification-gencode)
#   --ptxas-options=-v   -> ptxas "--verbose": prints registers / smem per kernel (#ptxas-options-verbose)
#   --compiler-options   -> forwarded to the host compiler (#compiler-options-options-xcompiler)
#   --shared             -> "Generate a shared library during linking" (#shared-shared)
# Compute capabilities: T4 = 7.5, L4 = 8.9 (https://developer.nvidia.com/cuda-gpus).

NVCC     ?= nvcc
GENCODE  ?= --generate-code arch=compute_75,code=sm_75 \
            '--generate-code=arch=compute_89,code=[sm_89,compute_89]'
HOSTOPTS := -fPIC,-Wall,-Wextra
NVFLAGS  := --shared -O3 -std=c++17 $(GENCODE) --compiler-options $(HOSTOPTS) --ptxas-options=-v
TARGET   := libredukt_gpu.so
SRC      := mamba_cuda.cu

.PHONY: all clean info

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVFLAGS) -o $@ $<
	@echo "[OK] Built $(TARGET)"
	@ls -lh $(TARGET)

clean:
	rm -f $(TARGET)

info:
	@echo "NVCC:    $(shell $(NVCC) --version | tail -1)"
	@echo "GENCODE: $(GENCODE)"
	@nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"
