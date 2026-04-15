package server

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// GenerateModelCard creates a Markdown model card for a GGUF model. OPT-129.
func GenerateModelCard(modelPath string, tokPerSec float64, latencyMs float64) string {
	info, _ := os.Stat(modelPath)
	sizeMB := float64(0)
	if info != nil {
		sizeMB = float64(info.Size()) / 1e6
	}

	name := strings.TrimSuffix(filepath.Base(modelPath), filepath.Ext(modelPath))

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("# Model Card: %s\n\n", name))
	sb.WriteString(fmt.Sprintf("**Generated:** %s\n\n", time.Now().Format("2006-01-02")))
	sb.WriteString("## Overview\n\n")
	sb.WriteString(fmt.Sprintf("| Property | Value |\n"))
	sb.WriteString(fmt.Sprintf("|---|---|\n"))
	sb.WriteString(fmt.Sprintf("| **File** | `%s` |\n", filepath.Base(modelPath)))
	sb.WriteString(fmt.Sprintf("| **Size** | %.1f MB |\n", sizeMB))
	sb.WriteString(fmt.Sprintf("| **Format** | %s |\n", strings.ToUpper(strings.TrimPrefix(filepath.Ext(modelPath), "."))))

	if tokPerSec > 0 {
		sb.WriteString("\n## Performance\n\n")
		sb.WriteString(fmt.Sprintf("| Metric | Value |\n"))
		sb.WriteString(fmt.Sprintf("|---|---|\n"))
		sb.WriteString(fmt.Sprintf("| **Tokens/sec** | %.1f |\n", tokPerSec))
		sb.WriteString(fmt.Sprintf("| **Latency** | %.1f ms |\n", latencyMs))
	}

	sb.WriteString("\n## Usage\n\n")
	sb.WriteString("```bash\n")
	sb.WriteString(fmt.Sprintf("infergo serve --model llm:%s --provider cuda\n", modelPath))
	sb.WriteString("```\n")

	sb.WriteString("\n## Limitations\n\n")
	sb.WriteString("- Model may generate inaccurate or biased content\n")
	sb.WriteString("- Not suitable for medical, legal, or safety-critical applications without human review\n")
	sb.WriteString("- Performance varies by task and prompt quality\n")

	return sb.String()
}
