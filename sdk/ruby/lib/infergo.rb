# infergo Ruby SDK — FFI bindings to libinfer_api.so
require 'ffi'

module Infergo
  extend FFI::Library

  lib_path = ENV['INFERGO_LIB_DIR'] ? File.join(ENV['INFERGO_LIB_DIR'], 'libinfer_api.so') : 'infer_api'
  ffi_lib lib_path

  # Error
  attach_function :infer_last_error_string, [], :string

  # LLM
  attach_function :infer_llm_create, [:string, :int, :int, :int, :int], :pointer
  attach_function :infer_llm_destroy, [:pointer], :void
  attach_function :infer_llm_vocab_size, [:pointer], :int
  attach_function :infer_llm_tokenize, [:pointer, :string, :int, :pointer, :int], :int
  attach_function :infer_llm_generate, [:pointer, :pointer, :int, :int, :float, :float, :string, :pointer, :pointer, :pointer, :int, :pointer], :int

  # Session / Tokenizer
  attach_function :infer_session_create, [:string, :int], :pointer
  attach_function :infer_session_load, [:pointer, :string], :int
  attach_function :infer_session_destroy, [:pointer], :void
  attach_function :infer_tokenizer_load, [:string], :pointer
  attach_function :infer_tokenizer_destroy, [:pointer], :void
  attach_function :infer_embed_pipeline, [:pointer, :pointer, :string, :pointer, :int], :int

  # VectorDB
  attach_function :infer_vectordb_create, [:int, :int, :int], :pointer
  attach_function :infer_vectordb_insert, [:pointer, :int64, :pointer, :string], :int
  attach_function :infer_vectordb_search, [:pointer, :pointer, :int, :int, :string, :pointer, :pointer, :int], :int
  attach_function :infer_vectordb_size, [:pointer], :int
  attach_function :infer_vectordb_free, [:pointer], :void

  # BM25
  attach_function :infer_bm25_create, [:float, :float], :pointer
  attach_function :infer_bm25_insert, [:pointer, :int64, :string], :void
  attach_function :infer_bm25_search, [:pointer, :string, :int, :pointer, :pointer, :int], :int
  attach_function :infer_bm25_size, [:pointer], :int
  attach_function :infer_bm25_free, [:pointer], :void

  class LLM
    def initialize(model_path, gpu_layers: -1, ctx_size: 4096, n_seq_max: 1, n_batch: 2048)
      @handle = Infergo.infer_llm_create(model_path, gpu_layers, ctx_size, n_seq_max, n_batch)
      raise "Failed to load: #{Infergo.infer_last_error_string}" if @handle.null?
      ObjectSpace.define_finalizer(self, self.class._release(@handle))
    end

    def self._release(handle)
      handle_ref = handle
      proc { Infergo.infer_llm_destroy(handle_ref) unless handle_ref.null? }
    end

    def vocab_size
      Infergo.infer_llm_vocab_size(@handle)
    end

    def tokenize(text, add_bos: true)
      out = FFI::MemoryPointer.new(:int, 4096)
      n = Infergo.infer_llm_tokenize(@handle, text, add_bos ? 1 : 0, out, 4096)
      raise "Tokenize failed" if n < 0
      out.read_array_of_int(n)
    end

    def generate(prompt, max_tokens: 128, temperature: 0.7, top_p: 0.9, grammar: nil)
      tokens = tokenize(prompt)
      tok_ptr = FFI::MemoryPointer.new(:int, tokens.length)
      tok_ptr.write_array_of_int(tokens)
      buf = FFI::MemoryPointer.new(:char, 32768)
      gen = FFI::MemoryPointer.new(:int)
      rc = Infergo.infer_llm_generate(@handle, tok_ptr, tokens.length, max_tokens,
                                       temperature, top_p, grammar, nil, nil, buf, 32768, gen)
      raise "Generate failed: #{Infergo.infer_last_error_string}" if rc < 0
      buf.read_string
    end

    def close
      Infergo.infer_llm_destroy(@handle) if @handle && !@handle.null?
      @handle = nil
    end
  end

  class Embedding
    def initialize(model_path, tokenizer_path, provider: "cpu", device_id: 0)
      @session = Infergo.infer_session_create(provider, device_id)
      raise "Session failed" if @session.null?
      rc = Infergo.infer_session_load(@session, model_path)
      raise "Load failed: #{Infergo.infer_last_error_string}" if rc != 0
      @tokenizer = Infergo.infer_tokenizer_load(tokenizer_path)
      raise "Tokenizer failed" if @tokenizer.null?
    end

    def embed(text)
      out = FFI::MemoryPointer.new(:float, 2048)
      dim = Infergo.infer_embed_pipeline(@session, @tokenizer, text, out, 2048)
      raise "Embed failed" if dim < 0
      out.read_array_of_float(dim)
    end

    def close
      Infergo.infer_tokenizer_destroy(@tokenizer) if @tokenizer
      Infergo.infer_session_destroy(@session) if @session
      @tokenizer = @session = nil
    end
  end

  class VectorDB
    def initialize(dim: 384, m: 16, ef_construction: 200)
      @handle = Infergo.infer_vectordb_create(dim, m, ef_construction)
      @dim = dim
    end

    def insert(id, vector, metadata: "")
      vec = FFI::MemoryPointer.new(:float, vector.length)
      vec.write_array_of_float(vector)
      Infergo.infer_vectordb_insert(@handle, id, vec, metadata)
    end

    def search(query, k: 10, ef_search: 50)
      q = FFI::MemoryPointer.new(:float, query.length)
      q.write_array_of_float(query)
      ids = FFI::MemoryPointer.new(:int64, k)
      dists = FFI::MemoryPointer.new(:float, k)
      n = Infergo.infer_vectordb_search(@handle, q, k, ef_search, nil, ids, dists, k)
      (0...n).map { |i| { id: ids.get_int64(i * 8), distance: dists.get_float32(i * 4) } }
    end

    def size; Infergo.infer_vectordb_size(@handle); end
    def close; Infergo.infer_vectordb_free(@handle) if @handle; @handle = nil; end
  end

  class BM25
    def initialize(k1: 1.2, b: 0.75)
      @handle = Infergo.infer_bm25_create(k1, b)
    end

    def insert(id, text); Infergo.infer_bm25_insert(@handle, id, text); end

    def search(query, k: 10)
      ids = FFI::MemoryPointer.new(:int64, k)
      scores = FFI::MemoryPointer.new(:float, k)
      n = Infergo.infer_bm25_search(@handle, query, k, ids, scores, k)
      (0...n).map { |i| { id: ids.get_int64(i * 8), score: scores.get_float32(i * 4) } }
    end

    def size; Infergo.infer_bm25_size(@handle); end
    def close; Infergo.infer_bm25_free(@handle) if @handle; @handle = nil; end
  end
end
