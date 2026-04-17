{:ok, llm} = Infergo.llm_create(System.argv() |> List.first() || "model.gguf")
{:ok, text} = Infergo.generate(llm, "What is Elixir?", max_tokens: 64)
IO.puts(text)
