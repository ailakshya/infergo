defmodule Infergo.MixProject do
  use Mix.Project

  def project do
    [
      app: :infergo,
      version: "1.0.0",
      elixir: "~> 1.14",
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_targets: ["all"],
      make_clean: ["clean"],
      deps: [{:elixir_make, "~> 0.7", runtime: false}]
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end
end
