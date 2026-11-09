# Redukt GPU CUDA Plugin — Build for Linux (Kaggle T4)
# Usage: make          → build libredukt_gpu.so
#        make test      → build + run Python validation
#        make benchmark → build + run throughput benchmark
#        make clean     → remove build artifacts

NVCC       ?= nvcc
ARCH       ?= sm_75
CUDA_FLAGS := -shared -O3 -arch=$(ARCH) --compiler-options '-fPIC' -Xcompiler '-Wall'
TARGET     := libredukt_gpu.so
SRC        := mamba_cuda.cu

.PHONY: all test benchmark clean info

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CUDA_FLAGS) -o $@ $<
	@echo "[OK] Built $(TARGET) for $(ARCH)"
	@ls -lh $(TARGET)

test: $(TARGET)
	python3 ../scripts/redukt-gpu-cuda-test.py --plugin ./$(TARGET)

benchmark: $(TARGET)
	python3 ../scripts/redukt-gpu-cuda-test.py --plugin ./$(TARGET) --benchmark

clean:
	rm -f $(TARGET)

info:
	@echo "NVCC:    $(shell $(NVCC) --version | head -1)"
	@echo "ARCH:    $(ARCH)"
	@echo "TARGET:  $(TARGET)"
	@nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"
