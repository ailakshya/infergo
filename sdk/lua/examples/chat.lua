local infergo = require("infergo")

local llm = infergo.LLM(arg[1] or "model.gguf")
print(llm:generate("What is Lua?", 64))
llm:close()
