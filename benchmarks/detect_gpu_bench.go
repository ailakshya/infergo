// Pure GPU detection benchmark — raw inference, no HTTP
package main

import (
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/ailakshya/infergo/torch"
)

const modelPath = "/home/lakshya/cgo/models/yolo11n.torchscript.pt"

func main() {
	const runs = 30
	const warmup = 10

	sess, err := torch.NewSession("cuda", 0)
	if err != nil { fmt.Fprintf(os.Stderr, "Session: %v\n", err); os.Exit(1) }
	defer sess.Close()

	if err := sess.Load(modelPath); err != nil { fmt.Fprintf(os.Stderr, "Load: %v\n", err); os.Exit(1) }

	jpeg := makeJPEG()
	fmt.Printf("TorchScript GPU detect (in-process) | %d bytes JPEG | %d runs\n\n", len(jpeg), runs)

	for i := 0; i < warmup; i++ { sess.DetectGPU(jpeg, 0.25, 0.45) }

	var times []float64
	for i := 0; i < runs; i++ {
		start := time.Now()
		_, err := sess.DetectGPU(jpeg, 0.25, 0.45)
		if err != nil { fmt.Printf("ERROR: %v\n", err); continue }
		times = append(times, float64(time.Since(start).Microseconds())/1000)
	}

	a := avg(times)
	m := mn(times)
	fmt.Printf("Avg: %.1fms | Min: %.1fms | %d RPS\n", a, m, int(1000/a))
	fmt.Printf("Python PyTorch ref: 2.8ms\n")
	fmt.Printf("Gap: %.1fms (%.1fx)\n", a-2.8, a/2.8)
}

func makeJPEG() []byte {
	out, err := exec.Command("python3", "-c",
		"import sys; from PIL import Image; import numpy as np; import io; "+
			"img=Image.fromarray(np.random.randint(0,255,(640,640,3),dtype=np.uint8)); "+
			"b=io.BytesIO(); img.save(b,format='JPEG',quality=85); sys.stdout.buffer.write(b.getvalue())").Output()
	if err == nil && len(out) > 100 { return out }
	d, _ := os.ReadFile("/tmp/bench_opt.jpg")
	return d
}

func avg(t []float64) float64 { s := 0.0; for _, v := range t { s += v }; return s / float64(len(t)) }
func mn(t []float64) float64 { m := t[0]; for _, v := range t[1:] { if v < m { m = v } }; return m }
