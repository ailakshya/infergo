import com.infergo.BM25;
import com.infergo.InfergoException;
import com.infergo.LLM;
import com.infergo.VectorDB;

import java.util.Scanner;

/**
 * Interactive chat example using the infergo Java SDK.
 *
 * Usage:
 *   gradle runChat --args="path/to/model.gguf"
 *
 * Or compile and run directly:
 *   javac -cp build/libs/java-0.1.0.jar examples/Chat.java
 *   java -Djava.library.path=build/jni -cp build/libs/java-0.1.0.jar:examples Chat model.gguf
 */
public class Chat {

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: Chat <model.gguf> [gpu_layers] [ctx_size]");
            System.err.println();
            System.err.println("  model.gguf   - path to a GGUF model file");
            System.err.println("  gpu_layers   - layers to offload to GPU (default: 99)");
            System.err.println("  ctx_size     - KV cache size in tokens (default: 4096)");
            System.exit(1);
        }

        String modelPath = args[0];
        int gpuLayers = args.length > 1 ? Integer.parseInt(args[1]) : 99;
        int ctxSize   = args.length > 2 ? Integer.parseInt(args[2]) : 4096;

        System.out.println("Loading model: " + modelPath);
        System.out.println("GPU layers: " + gpuLayers + ", context size: " + ctxSize);
        System.out.println();

        try (LLM llm = new LLM(modelPath, gpuLayers, ctxSize, 1, 512)) {
            System.out.println("Model loaded. Vocab size: " + llm.vocabSize());
            System.out.println("Type your message (or 'quit' to exit):");
            System.out.println("─".repeat(60));

            Scanner scanner = new Scanner(System.in);
            while (true) {
                System.out.print("\nYou: ");
                String input = scanner.nextLine().trim();
                if (input.isEmpty()) continue;
                if (input.equalsIgnoreCase("quit") || input.equalsIgnoreCase("exit")) {
                    break;
                }

                try {
                    long start = System.nanoTime();
                    String response = llm.generate(input, 256, 0.7f, 0.9f);
                    long elapsed = System.nanoTime() - start;

                    System.out.println("\nAssistant: " + response);
                    System.out.printf("  [%.1f ms]%n", elapsed / 1_000_000.0);
                } catch (InfergoException e) {
                    System.err.println("Error: " + e.getMessage());
                }
            }

            System.out.println("\nGoodbye!");
        }

        // ── BM25 demo ───────────────────────────────────────────────────────
        System.out.println("\n── BM25 Full-Text Search Demo ──");
        try (BM25 index = new BM25()) {
            index.insert(1, "The quick brown fox jumps over the lazy dog");
            index.insert(2, "A fast red car drove past the sleeping cat");
            index.insert(3, "Brown bears hibernate during winter months");

            BM25.SearchResult[] results = index.search("brown fox", 3);
            System.out.println("Query: 'brown fox'");
            for (BM25.SearchResult r : results) {
                System.out.printf("  id=%d score=%.4f%n", r.id, r.score);
            }
            System.out.println("Index size: " + index.size());
        }
    }
}
