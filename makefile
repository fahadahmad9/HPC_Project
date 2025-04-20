# CUDA compiler
CC = nvcc

# Compiler flags
CFLAGS = -O2 -g -arch=sm_70   # Specify compute capability, e.g., sm_70 for Volta

# Linker flags (uncomment if using CUDA runtime or cuBLAS explicitly)
LDFLAGS = -lcudart -lcublas

# Executable name and source file
EXE = nn4.exe
SRC = V4.cu

# Default target
all: $(EXE) run

# Build rule
$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $(EXE) $(SRC) $(LDFLAGS)

# Run target
run: $(EXE)
	./$(EXE)

# Clean target
clean:
	rm -f $(EXE)
