# Build on the board (native aarch64). Two artifacts:
#   fl_run                    CLI (bench + single image)
#   libfacelandmarker_npu.so  shared library for FFI callers (Python ctypes...)
#
# VIPLITE points at the ai-sdk VIPLite v2.0 for aarch64 (headers + libs):
#   git clone --depth 1 https://github.com/ZIFENG278/ai-sdk.git ~/ai-sdk
VIPLITE ?= $(HOME)/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0

CC      ?= gcc
CFLAGS  += -O3 -mcpu=cortex-a76 -Wall -Wextra -fPIC -I$(VIPLITE)/inc
LDLIBS  += -L$(VIPLITE) -lNBGlinker -lVIPhal -lm

SRC = src/vipnet.c src/facelandmarker.c

all: fl_run libfacelandmarker_npu.so

fl_run: $(SRC) src/main.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

libfacelandmarker_npu.so: $(SRC)
	$(CC) $(CFLAGS) -shared -o $@ $^ $(LDLIBS)

clean:
	rm -f fl_run libfacelandmarker_npu.so

.PHONY: all clean
