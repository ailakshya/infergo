require_relative '../lib/infergo'

llm = Infergo::LLM.new(ARGV[0] || "model.gguf")
puts llm.generate("What is Ruby?", max_tokens: 64)
llm.close
