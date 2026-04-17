INFERGO_ROOT ?= ../..
INFERGO_INCLUDE ?= $(INFERGO_ROOT)/cpp/include
INFERGO_LIB ?= $(INFERGO_ROOT)/build

ERLANG_PATH = $(shell erl -eval 'io:format("~s", [lists:concat([code:root_dir(), "/erts-", erlang:system_info(version), "/include"])])' -s init stop -noshell)
ERL_INTERFACE_INCLUDE = $(shell erl -eval 'io:format("~s", [code:lib_dir(erl_interface, include)])' -s init stop -noshell)

CC ?= gcc
CFLAGS = -O2 -fPIC -shared \
         -I$(ERLANG_PATH) \
         -I$(ERL_INTERFACE_INCLUDE) \
         -I$(INFERGO_INCLUDE)

LDFLAGS = -L$(INFERGO_LIB) -linfer_api -Wl,-rpath,$(INFERGO_LIB)

PRIV_DIR = priv
NIF = $(PRIV_DIR)/infergo_nif.so

all: $(NIF)

$(NIF): c_src/infergo_nif.c
	@mkdir -p $(PRIV_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -rf $(PRIV_DIR)/infergo_nif.so

.PHONY: all clean
