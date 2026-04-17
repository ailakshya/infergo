INFERGO_ROOT ?= ../..
CFLAGS = -O2 -I$(INFERGO_ROOT)/cpp/include
EMCC_FLAGS = -s EXPORTED_RUNTIME_METHODS='["allocateUTF8","UTF8ToString","FS"]' \
             -s EXPORTED_FUNCTIONS='["_malloc","_free"]' \
             -s ALLOW_MEMORY_GROWTH=1 \
             -s MODULARIZE=1 \
             -s EXPORT_ES6=0 \
             -s INITIAL_MEMORY=134217728 \
             -s STACK_SIZE=1048576

all: infergo.js

infergo.js: src/infergo_wasm.c
	emcc $(CFLAGS) $(EMCC_FLAGS) -o $@ $< \
		-L$(INFERGO_ROOT)/build -linfer_api

clean:
	rm -f infergo.js infergo.wasm
