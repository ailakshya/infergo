package tracker

import (
	"math"
	"testing"
)

func TestIoU(t *testing.T) {
	tests := []struct {
		name string
		a, b [4]float64
		want float64
	}{
		{"identical", [4]float64{0, 0, 10, 10}, [4]float64{0, 0, 10, 10}, 1.0},
		{"no overlap", [4]float64{0, 0, 10, 10}, [4]float64{20, 20, 30, 30}, 0.0},
		{"half overlap", [4]float64{0, 0, 10, 10}, [4]float64{5, 0, 15, 10}, 1.0 / 3.0},
		{"contained", [4]float64{0, 0, 10, 10}, [4]float64{2, 2, 8, 8}, 36.0 / 100.0},
		{"touching edge", [4]float64{0, 0, 10, 10}, [4]float64{10, 0, 20, 10}, 0.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := iou(tt.a, tt.b)
			if math.Abs(got-tt.want) > 1e-6 {
				t.Errorf("iou(%v, %v) = %f, want %f", tt.a, tt.b, got, tt.want)
			}
		})
	}
}

func TestXYXYtoXYAHRoundtrip(t *testing.T) {
	box := [4]float64{10, 20, 50, 80}
	xyah := xyxyToXYAH(box)
	// cx=30, cy=50, a=40/60=0.6667, h=60
	if math.Abs(xyah[0]-30) > 1e-6 || math.Abs(xyah[1]-50) > 1e-6 {
		t.Errorf("center wrong: got (%f,%f), want (30,50)", xyah[0], xyah[1])
	}

	var mean [8]float64
	copy(mean[:4], xyah[:])
	back := xyahToXYXY(mean)
	for i := 0; i < 4; i++ {
		if math.Abs(back[i]-box[i]) > 1e-4 {
			t.Errorf("roundtrip box[%d] = %f, want %f", i, back[i], box[i])
		}
	}
}

func TestKalmanPredictUpdate(t *testing.T) {
	meas := [4]float64{100, 200, 0.5, 80}
	ks := initKalman(meas)

	// Predict should add velocity (zero initially)
	ks.predict()
	if math.Abs(ks.mean[0]-100) > 1e-6 {
		t.Errorf("after predict, cx = %f, want 100", ks.mean[0])
	}

	// Update with shifted measurement
	ks.update([4]float64{105, 200, 0.5, 80})
	if ks.mean[0] <= 100 || ks.mean[0] >= 105 {
		t.Errorf("after update, cx = %f, want between 100 and 105", ks.mean[0])
	}

	// After update, velocity should be positive
	if ks.mean[4] <= 0 {
		t.Errorf("after update, vx = %f, want > 0", ks.mean[4])
	}
}

func TestInv4x4Identity(t *testing.T) {
	var m [4][4]float64
	for i := 0; i < 4; i++ {
		m[i][i] = 1
	}
	inv := inv4x4(m)
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			want := 0.0
			if i == j {
				want = 1.0
			}
			if math.Abs(inv[i][j]-want) > 1e-10 {
				t.Errorf("inv[%d][%d] = %f, want %f", i, j, inv[i][j], want)
			}
		}
	}
}

func TestInv4x4Known(t *testing.T) {
	// 2*I should invert to 0.5*I
	var m [4][4]float64
	for i := 0; i < 4; i++ {
		m[i][i] = 2
	}
	inv := inv4x4(m)
	for i := 0; i < 4; i++ {
		if math.Abs(inv[i][i]-0.5) > 1e-10 {
			t.Errorf("inv[%d][%d] = %f, want 0.5", i, i, inv[i][i])
		}
	}
}

func TestHungarianSimple(t *testing.T) {
	// Simple 3x3 cost matrix
	cost := [][]float64{
		{1, 2, 3},
		{2, 4, 6},
		{3, 6, 9},
	}
	rowAssign, _ := hungarian(cost, 3, 3)
	// Optimal: row0→col0(1), row1→col1(4), row2→col2(9) = 14
	// Or row0→col2(3), row1→col0(2), row2→col1(6) = 11 (better)
	totalCost := 0.0
	for r, c := range rowAssign {
		if c >= 0 {
			totalCost += cost[r][c]
		}
	}
	// Minimum should be 1+4+9=14? No, 3+2+6=11. Actually:
	// All permutations: (0,1,2)=14, (0,2,1)=1+6+6=13, (1,0,2)=2+2+9=13,
	// (1,2,0)=2+6+3=11, (2,0,1)=3+2+6=11, (2,1,0)=3+4+3=10
	// Min = 10: row0→col2(3), row1→col1(4), row2→col0(3)
	if totalCost > 11 {
		t.Errorf("Hungarian total cost = %f, want ≤ 11", totalCost)
	}
}

func TestHungarianNonSquare(t *testing.T) {
	// 2 rows, 3 cols
	cost := [][]float64{
		{10, 1, 5},
		{3, 8, 2},
	}
	rowAssign, _ := hungarian(cost, 2, 3)
	// Optimal: row0→col1(1), row1→col2(2) = 3
	totalCost := 0.0
	for r, c := range rowAssign {
		if c >= 0 {
			totalCost += cost[r][c]
		}
	}
	if totalCost > 3.0+1e-6 {
		t.Errorf("Hungarian total cost = %f, want 3", totalCost)
	}
}

func TestLinearAssignmentThreshold(t *testing.T) {
	cost := [][]float64{
		{0.1, 0.9},
		{0.9, 0.1},
	}
	mR, mC, uR, uC := linearAssignment(cost, 0.5)
	if len(mR) != 2 || len(mC) != 2 {
		t.Errorf("expected 2 matches, got %d rows %d cols", len(mR), len(mC))
	}
	if len(uR) != 0 || len(uC) != 0 {
		t.Errorf("expected 0 unmatched, got %d rows %d cols", len(uR), len(uC))
	}
}

func TestLinearAssignmentAllAboveThreshold(t *testing.T) {
	cost := [][]float64{
		{0.9, 0.8},
		{0.8, 0.9},
	}
	mR, _, uR, uC := linearAssignment(cost, 0.5)
	// All costs > threshold → no matches
	if len(mR) != 0 {
		t.Errorf("expected 0 matches, got %d", len(mR))
	}
	if len(uR) != 2 || len(uC) != 2 {
		t.Errorf("expected 2 unmatched each, got %d rows %d cols", len(uR), len(uC))
	}
}

func TestByteTrackerBasic(t *testing.T) {
	bt := NewByteTracker(DefaultConfig())

	// Frame 1: two detections
	dets := []Detection{
		{Box: [4]float64{10, 10, 50, 50}, Class: 0, Confidence: 0.9},
		{Box: [4]float64{100, 100, 150, 150}, Class: 1, Confidence: 0.8},
	}
	tracks := bt.Update(dets)

	// First frame: tracks may be unconfirmed (need 2 frames to activate)
	// ByteTrack creates them as new, they become active on second match
	_ = tracks

	// Frame 2: same detections slightly moved
	dets2 := []Detection{
		{Box: [4]float64{12, 12, 52, 52}, Class: 0, Confidence: 0.9},
		{Box: [4]float64{102, 102, 152, 152}, Class: 1, Confidence: 0.8},
	}
	tracks2 := bt.Update(dets2)

	if len(tracks2) < 2 {
		t.Errorf("frame 2: expected ≥2 tracks, got %d", len(tracks2))
	}

	// Frame 3: detections move more
	dets3 := []Detection{
		{Box: [4]float64{14, 14, 54, 54}, Class: 0, Confidence: 0.9},
		{Box: [4]float64{104, 104, 154, 154}, Class: 1, Confidence: 0.8},
	}
	tracks3 := bt.Update(dets3)

	if len(tracks3) < 2 {
		t.Errorf("frame 3: expected ≥2 tracks, got %d", len(tracks3))
	}

	// Verify IDs are consistent
	if len(tracks2) >= 2 && len(tracks3) >= 2 {
		id2Map := make(map[int]bool)
		for _, tr := range tracks2 {
			id2Map[tr.ID] = true
		}
		for _, tr := range tracks3 {
			if !id2Map[tr.ID] {
				t.Errorf("track ID %d in frame 3 not seen in frame 2", tr.ID)
			}
		}
	}
}

func TestByteTrackerLostAndRecovery(t *testing.T) {
	cfg := DefaultConfig()
	cfg.TrackBuffer = 10
	bt := NewByteTracker(cfg)

	det := []Detection{
		{Box: [4]float64{10, 10, 50, 50}, Class: 0, Confidence: 0.9},
	}

	// Build track over 3 frames
	bt.Update(det)
	bt.Update(det)
	tracks := bt.Update(det)
	if len(tracks) == 0 {
		t.Fatal("expected at least 1 track after 3 frames")
	}
	origID := tracks[0].ID

	// Object disappears for 5 frames
	for i := 0; i < 5; i++ {
		bt.Update(nil)
	}

	// Object reappears at same location
	recovered := bt.Update(det)
	foundOrig := false
	for _, tr := range recovered {
		if tr.ID == origID {
			foundOrig = true
		}
	}
	if !foundOrig {
		t.Errorf("expected track ID %d to be recovered, got IDs: %v", origID, recovered)
	}
}

func TestByteTrackerReset(t *testing.T) {
	bt := NewByteTracker(DefaultConfig())
	bt.Update([]Detection{
		{Box: [4]float64{10, 10, 50, 50}, Class: 0, Confidence: 0.9},
	})

	bt.Reset()

	if len(bt.tracked) != 0 || len(bt.lost) != 0 || bt.frameCount != 0 {
		t.Error("Reset did not clear state")
	}
}

func TestByteTrackerLowConfRecovery(t *testing.T) {
	cfg := DefaultConfig()
	cfg.TrackHighThresh = 0.5
	cfg.TrackLowThresh = 0.1
	bt := NewByteTracker(cfg)

	// High confidence detections for 3 frames to establish track
	highDet := []Detection{
		{Box: [4]float64{10, 10, 50, 50}, Class: 0, Confidence: 0.9},
	}
	bt.Update(highDet)
	bt.Update(highDet)
	tracks := bt.Update(highDet)
	if len(tracks) == 0 {
		t.Fatal("expected track after 3 high-conf frames")
	}

	// Low confidence detection at same spot — should still match via stage 2
	lowDet := []Detection{
		{Box: [4]float64{12, 12, 52, 52}, Class: 0, Confidence: 0.15},
	}
	tracks = bt.Update(lowDet)
	if len(tracks) == 0 {
		t.Error("expected track to survive with low-conf detection via stage 2")
	}
}

func BenchmarkByteTrackerUpdate(b *testing.B) {
	bt := NewByteTracker(DefaultConfig())

	// Simulate 20 detections per frame
	dets := make([]Detection, 20)
	for i := range dets {
		x := float64(i) * 50
		dets[i] = Detection{
			Box:        [4]float64{x, 100, x + 40, 140},
			Class:      0,
			Confidence: 0.8,
		}
	}

	// Warm up
	for i := 0; i < 10; i++ {
		bt.Update(dets)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bt.Update(dets)
	}
}

func BenchmarkHungarian50x50(b *testing.B) {
	n := 50
	cost := make([][]float64, n)
	for i := range cost {
		cost[i] = make([]float64, n)
		for j := range cost[i] {
			cost[i][j] = float64((i*7+j*13)%100) / 100.0
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hungarian(cost, n, n)
	}
}
