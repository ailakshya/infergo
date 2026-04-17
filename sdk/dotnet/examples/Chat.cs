// infergo .NET SDK — Chat example
//
// Build:
//   dotnet run --project examples/Chat.csproj -- path/to/model.gguf
//
// Requires libinfer_api.so (Linux) or infer_api.dll (Windows) on LD_LIBRARY_PATH.

using Infergo;

if (args.Length < 1)
{
    Console.Error.WriteLine("Usage: Chat <model.gguf> [gpu_layers]");
    return 1;
}

string modelPath = args[0];
int gpuLayers = args.Length > 1 ? int.Parse(args[1]) : 999;

Console.WriteLine($"Loading model: {modelPath}");

using var llm = new Llm(modelPath, gpuLayers: gpuLayers, ctxSize: 4096);
Console.WriteLine($"Model loaded. Vocab size: {llm.VocabSize}");
Console.WriteLine("Type a message (or 'quit' to exit).\n");

while (true)
{
    Console.Write("> ");
    string? input = Console.ReadLine();
    if (input == null || input.Equals("quit", StringComparison.OrdinalIgnoreCase))
        break;

    if (string.IsNullOrWhiteSpace(input))
        continue;

    // Build a simple chat prompt (Llama-3 format)
    string prompt = $"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";

    try
    {
        // Stream tokens as they are generated
        var (text, tokenCount) = llm.Generate(
            prompt,
            maxTokens: 512,
            temperature: 0.7f,
            topP: 0.9f,
            callback: (token, piece) =>
            {
                Console.Write(piece);
                return true; // continue generating
            });

        Console.WriteLine(); // newline after streamed output
        Console.WriteLine($"[{tokenCount} tokens generated]\n");
    }
    catch (InfergoException ex)
    {
        Console.Error.WriteLine($"Error: {ex.Message}");
    }
}

Console.WriteLine("Bye.");
return 0;
