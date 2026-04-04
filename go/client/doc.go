// Package client provides a typed Go client for the infergo inference server.
//
// Quick start:
//
//	c := client.New("http://localhost:9090", client.WithAPIKey("secret"))
//
//	resp, err := c.Chat(ctx, client.ChatRequest{
//	    Model:    "llama3-8b-q4",
//	    Messages: []client.Message{{Role: "user", Content: "Hello"}},
//	})
//
//	vec, err := c.Embed(ctx, client.EmbedRequest{
//	    Model: "all-MiniLM-L6-v2",
//	    Input: "semantic search query",
//	})
package client
