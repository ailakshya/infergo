package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

// chatCmd implements `infergo chat` — interactive terminal chat.
// Connects to a running infergo server or loads a model directly.
func chatCmd(args []string) {
	// Parse flags
	serverURL := "http://localhost:9090"
	model := "llm"
	system := ""

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--server", "-s":
			if i+1 < len(args) {
				serverURL = args[i+1]
				i++
			}
		case "--model", "-m":
			if i+1 < len(args) {
				model = args[i+1]
				i++
			}
		case "--system":
			if i+1 < len(args) {
				system = args[i+1]
				i++
			}
		case "--help", "-h":
			fmt.Println("Usage: infergo chat [options]")
			fmt.Println()
			fmt.Println("Options:")
			fmt.Println("  --server, -s URL    Server URL (default: http://localhost:9090)")
			fmt.Println("  --model, -m NAME    Model name (default: llm)")
			fmt.Println("  --system TEXT       System prompt")
			fmt.Println()
			fmt.Println("Commands:")
			fmt.Println("  /help               Show help")
			fmt.Println("  /clear              Clear conversation")
			fmt.Println("  /system <text>      Set system prompt")
			fmt.Println("  /model <name>       Switch model")
			fmt.Println("  /quit, /exit        Exit")
			return
		}
	}

	fmt.Printf("infergo chat — connected to %s (model: %s)\n", serverURL, model)
	fmt.Println("Type /help for commands, /quit to exit.")

	var messages []chatMsg
	if system != "" {
		messages = append(messages, chatMsg{Role: "system", Content: system})
	}

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB line buffer

	for {
		fmt.Print(">>> ")
		if !scanner.Scan() {
			break
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		// Handle commands
		if strings.HasPrefix(input, "/") {
			switch {
			case input == "/quit" || input == "/exit":
				fmt.Println("Bye!")
				return
			case input == "/clear":
				messages = messages[:0]
				if system != "" {
					messages = append(messages, chatMsg{Role: "system", Content: system})
				}
				fmt.Println("Conversation cleared.")
				continue
			case input == "/help":
				fmt.Println("Commands: /clear /system <text> /model <name> /quit")
				continue
			case strings.HasPrefix(input, "/system "):
				system = strings.TrimPrefix(input, "/system ")
				// Replace or add system message
				if len(messages) > 0 && messages[0].Role == "system" {
					messages[0].Content = system
				} else {
					messages = append([]chatMsg{{Role: "system", Content: system}}, messages...)
				}
				fmt.Printf("System prompt set: %s\n", system)
				continue
			case strings.HasPrefix(input, "/model "):
				model = strings.TrimPrefix(input, "/model ")
				fmt.Printf("Switched to model: %s\n", model)
				continue
			default:
				fmt.Printf("Unknown command: %s (try /help)\n", input)
				continue
			}
		}

		// Add user message
		messages = append(messages, chatMsg{Role: "user", Content: input})

		// Stream response
		response, err := streamChat(serverURL, model, messages)
		if err != nil {
			fmt.Printf("\nError: %s\n", err)
			continue
		}

		// Add assistant response to history
		messages = append(messages, chatMsg{Role: "assistant", Content: response})
		fmt.Println()
	}
}

type chatMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatReq struct {
	Model    string    `json:"model"`
	Messages []chatMsg `json:"messages"`
	Stream   bool      `json:"stream"`
}

// streamChat sends a streaming request and prints tokens as they arrive.
func streamChat(serverURL, model string, messages []chatMsg) (string, error) {
	reqBody, _ := json.Marshal(chatReq{
		Model:    model,
		Messages: messages,
		Stream:   true,
	})

	resp, err := http.Post(serverURL+"/v1/chat/completions", "application/json",
		strings.NewReader(string(reqBody)))
	if err != nil {
		return "", fmt.Errorf("connection failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	// Parse SSE stream
	var fullResponse strings.Builder
	reader := bufio.NewReader(resp.Body)

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(data), &chunk) == nil && len(chunk.Choices) > 0 {
			content := chunk.Choices[0].Delta.Content
			fmt.Print(content)
			fullResponse.WriteString(content)
		}
	}

	// If streaming didn't work (non-streaming response), try parsing as regular JSON
	if fullResponse.Len() == 0 {
		// Re-read body for non-streaming
		resp2, err := http.Post(serverURL+"/v1/chat/completions", "application/json",
			strings.NewReader(string(reqBody[:len(reqBody)-1])+`,"stream":false}`))
		if err == nil {
			defer resp2.Body.Close()
			var result struct {
				Choices []struct {
					Message struct {
						Content string `json:"content"`
					} `json:"message"`
				} `json:"choices"`
			}
			if json.NewDecoder(resp2.Body).Decode(&result) == nil && len(result.Choices) > 0 {
				content := result.Choices[0].Message.Content
				fmt.Print(content)
				fullResponse.WriteString(content)
			}
		}
	}

	return fullResponse.String(), nil
}
