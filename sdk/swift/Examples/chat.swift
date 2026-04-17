// chat.swift — Interactive chat example using the Infergo Swift SDK.
//
// Build:
//   swift build -Xlinker -L/path/to/infergo/build -Xcc -I/path/to/infergo/cpp/include
//
// Run:
//   .build/debug/chat /path/to/model.gguf

import Foundation
import Infergo

func main() throws {
    let args = CommandLine.arguments
    guard args.count >= 2 else {
        print("Usage: chat <model.gguf> [gpu_layers]")
        return
    }

    let modelPath = args[1]
    let gpuLayers: Int32 = args.count >= 3 ? Int32(args[2]) ?? 99 : 99

    print("Loading model: \(modelPath) ...")
    let llm = try LLM(path: modelPath, gpuLayers: gpuLayers, contextSize: 4096)
    print("Loaded. Vocab size: \(llm.vocabSize). Type 'quit' to exit.\n")

    // Simple multi-turn chat loop.
    var history = ""

    while true {
        print("You: ", terminator: "")
        guard let line = readLine(), !line.isEmpty else { continue }
        if line.lowercased() == "quit" { break }

        // Build a simple prompt with history context.
        history += "User: \(line)\nAssistant:"

        // Stream tokens to stdout as they arrive.
        let result = try llm.generate(
            prompt: history,
            maxTokens: 256,
            temperature: 0.7,
            topP: 0.9,
            callback: { _, piece in
                print(piece, terminator: "")
                fflush(stdout)
                return true
            }
        )
        print()  // newline after streaming

        // Append the assistant reply to history for context.
        history += " \(result.text)\n"
    }

    print("Bye!")
}

try main()
