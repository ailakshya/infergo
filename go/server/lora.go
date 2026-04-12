package server

// LoRA adapter types — hot-swap fine-tuned weights per request.

// LoRAAdapter represents a loaded LoRA adapter.
type LoRAAdapter struct {
	Name string `json:"name"`
	Path string `json:"path"`
	Rank int    `json:"rank,omitempty"`
}

// LoRARequest allows per-request adapter selection.
// Added to ChatCompletionRequest as optional field.
type LoRARequest struct {
	Adapter string  `json:"adapter,omitempty"` // adapter name to use
	Scale   float32 `json:"scale,omitempty"`   // LoRA scaling factor (default 1.0)
}

// LoRAModel is implemented by LLM models that support LoRA adapters.
type LoRAModel interface {
	Model
	LoadAdapter(name, path string) error
	UnloadAdapter(name string) error
	ListAdapters() []LoRAAdapter
}
