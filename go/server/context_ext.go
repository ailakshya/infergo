package server

// ContextExtension documents RoPE scaling configuration. OPT-125.
// llama.cpp handles this natively via model params — we just expose the CLI flags.

// RoPEScalingConfig holds context extension parameters.
type RoPEScalingConfig struct {
	Type      string  `json:"type"`       // "yarn", "ntk", "linear", "none"
	Factor    float64 `json:"factor"`     // scaling factor (e.g., 4.0 for 4x context)
	OrigCtx   int     `json:"orig_ctx"`   // original training context length
	TargetCtx int     `json:"target_ctx"` // target extended context length
}

// DefaultRoPEScaling returns no scaling (native context).
func DefaultRoPEScaling() RoPEScalingConfig {
	return RoPEScalingConfig{Type: "none", Factor: 1.0}
}

// YaRNScaling returns YaRN scaling config for context extension.
func YaRNScaling(origCtx, targetCtx int) RoPEScalingConfig {
	factor := float64(targetCtx) / float64(origCtx)
	return RoPEScalingConfig{
		Type:      "yarn",
		Factor:    factor,
		OrigCtx:   origCtx,
		TargetCtx: targetCtx,
	}
}

// NTKScaling returns NTK-aware scaling config.
func NTKScaling(origCtx, targetCtx int) RoPEScalingConfig {
	factor := float64(targetCtx) / float64(origCtx)
	return RoPEScalingConfig{
		Type:      "ntk",
		Factor:    factor,
		OrigCtx:   origCtx,
		TargetCtx: targetCtx,
	}
}
