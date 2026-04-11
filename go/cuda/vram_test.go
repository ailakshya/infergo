package cuda

import (
	"testing"
)

func TestVRAMTotal(t *testing.T) {
	total := VRAMTotal()
	if total == 0 {
		t.Skip("CUDA not available — skipping VRAM tests")
	}
	t.Logf("VRAM total: %d bytes (%.1f GiB)", total, float64(total)/(1024*1024*1024))
}

func TestVRAMFree(t *testing.T) {
	total := VRAMTotal()
	if total == 0 {
		t.Skip("CUDA not available — skipping VRAM tests")
	}
	free := VRAMFree()
	if free == 0 {
		t.Error("VRAMFree returned 0 on a machine with CUDA")
	}
	if free > total {
		t.Errorf("VRAMFree (%d) > VRAMTotal (%d)", free, total)
	}
	t.Logf("VRAM free: %d bytes (%.1f GiB)", free, float64(free)/(1024*1024*1024))
}

func TestVRAMUsedPct(t *testing.T) {
	total := VRAMTotal()
	if total == 0 {
		t.Skip("CUDA not available — skipping VRAM tests")
	}
	pct := VRAMUsedPct()
	if pct < 0 || pct > 100 {
		t.Errorf("VRAMUsedPct returned %d, want 0-100", pct)
	}
	t.Logf("VRAM used: %d%%", pct)
}

func BenchmarkVRAMQuery(b *testing.B) {
	if VRAMTotal() == 0 {
		b.Skip("CUDA not available")
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VRAMFree()
	}
}
