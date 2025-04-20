# Use nvcc instead of gcc for CUDA compilation
CC = nvcc
CFLAGS = -O2 -g

EXE = nn.exe
SRC = V4.cu

all: $(EXE) run

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $(EXE) $(SRC) -lm

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE)
