defmodule Infergo do
  @moduledoc "infergo Elixir SDK — NIF bindings to libinfer_api.so"
  @on_load :init

  def init do
    path = Path.join(:code.priv_dir(:infergo), "infergo_nif")
    :erlang.load_nif(String.to_charlist(path), 0)
  end

  # NIFs
  def llm_create_nif(_path, _gpu, _ctx, _seq, _batch), do: :erlang.nif_error(:nif_not_loaded)
  def llm_generate_nif(_llm, _prompt, _max, _temp), do: :erlang.nif_error(:nif_not_loaded)

  # Public API
  def llm_create(path, opts \\ []) do
    gpu = Keyword.get(opts, :gpu_layers, -1)
    ctx = Keyword.get(opts, :ctx_size, 4096)
    seq = Keyword.get(opts, :n_seq_max, 1)
    batch = Keyword.get(opts, :n_batch, 2048)
    llm_create_nif(String.to_charlist(path), gpu, ctx, seq, batch)
  end

  def generate(llm, prompt, opts \\ []) do
    max = Keyword.get(opts, :max_tokens, 128)
    temp = Keyword.get(opts, :temperature, 0.7)
    llm_generate_nif(llm, String.to_charlist(prompt), max, temp / 1)
  end
end
