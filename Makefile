# Makefile for standalone C reduced CNN build (does not touch parent project)

CC ?= gcc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter
# Link against math library for exp() used in softmax
LDFLAGS ?= -lm

TARGET = reduced_cnn_demo
SRC_DEMO = reduced_cnn.c main.c
OBJ_DEMO = $(SRC_DEMO:.c=.o)

APP_TARGET = cnn_app
SRC_APP = reduced_cnn.c training.c cnn_app.c
OBJ_APP = $(SRC_APP:.c=.o)

all: $(TARGET) $(APP_TARGET)

$(TARGET): $(OBJ_DEMO)
	$(CC) $(CFLAGS) -o $@ $(OBJ_DEMO) $(LDFLAGS)

$(APP_TARGET): gen_headers $(OBJ_APP)
	$(CC) $(CFLAGS) -o $@ $(OBJ_APP) $(LDFLAGS)

gen_headers:
	@echo "[gen] Generating image headers" && /home/ameyk/windows_d_drive/PSU/new_GitHub_repos/CNN_hand_written_digit/.venv/bin/python gen_image_headers.py || echo "Header generation failed"

reduced_cnn.o: reduced_cnn.c reduced_cnn.h
	$(CC) $(CFLAGS) -c reduced_cnn.c -o reduced_cnn.o
main.o: main.c reduced_cnn.h
	$(CC) $(CFLAGS) -c main.c -o main.o
training.o: training.c reduced_cnn.h dataset_arrays.h
	$(CC) $(CFLAGS) -c training.c -o training.o
cnn_app.o: cnn_app.c reduced_cnn.h dataset_arrays.h
	$(CC) $(CFLAGS) -c cnn_app.c -o cnn_app.o

clean:
	rm -f $(OBJ_DEMO) $(OBJ_APP) $(TARGET) $(APP_TARGET)

distclean: clean
	rm -rf generated_images dataset_arrays.h trained_reduced_cnn.bin

.PHONY: all clean distclean gen_headers

# --- Minimal pure inference build (no args) ---
# Generate weights.h from trained_reduced_cnn.bin
weights.h: export_weights.py trained_reduced_cnn.bin
	python3 export_weights.py trained_reduced_cnn.bin weights.h

# Build the tiny pure-C inference program
mb_infer_pure: mb_infer_pure.c weights.h
	$(CC) -O2 -s -std=c11 -o $@ mb_infer_pure.c

# Convenience target: build weights and run inference once
run_infer: weights.h mb_infer_pure
	./mb_infer_pure

.PHONY: all clean run_infer
