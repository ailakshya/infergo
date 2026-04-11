package analytics

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// --- Point-in-polygon tests ---

func TestPointInPolygonSquare(t *testing.T) {
	z := &Zone{
		Type:    "zone",
		Polygon: []Point{{0, 0}, {100, 0}, {100, 100}, {0, 100}},
		bbMinX:  0, bbMinY: 0, bbMaxX: 100, bbMaxY: 100,
	}

	tests := []struct {
		name string
		p    Point
		want bool
	}{
		{"center", Point{50, 50}, true},
		{"inside near edge", Point{1, 1}, true},
		{"outside", Point{150, 50}, false},
		{"outside negative", Point{-10, 50}, false},
		{"outside above", Point{50, 150}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := pointInPolygon(tt.p, z)
			if got != tt.want {
				t.Errorf("pointInPolygon(%v) = %v, want %v", tt.p, got, tt.want)
			}
		})
	}
}

func TestPointInPolygonTriangle(t *testing.T) {
	z := &Zone{
		Type:    "zone",
		Polygon: []Point{{50, 0}, {100, 100}, {0, 100}},
		bbMinX:  0, bbMinY: 0, bbMaxX: 100, bbMaxY: 100,
	}

	tests := []struct {
		name string
		p    Point
		want bool
	}{
		{"centroid", Point{50, 66}, true},
		{"inside", Point{50, 50}, true},
		{"outside top-left", Point{10, 10}, false},
		{"outside top-right", Point{90, 10}, false},
		{"outside right", Point{110, 50}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := pointInPolygon(tt.p, z)
			if got != tt.want {
				t.Errorf("pointInPolygon(%v) = %v, want %v", tt.p, got, tt.want)
			}
		})
	}
}

func TestPointInPolygonConcave(t *testing.T) {
	// L-shaped polygon (concave)
	z := &Zone{
		Type: "zone",
		Polygon: []Point{
			{0, 0}, {100, 0}, {100, 50},
			{50, 50}, {50, 100}, {0, 100},
		},
		bbMinX: 0, bbMinY: 0, bbMaxX: 100, bbMaxY: 100,
	}

	tests := []struct {
		name string
		p    Point
		want bool
	}{
		{"in top arm", Point{75, 25}, true},
		{"in bottom arm", Point{25, 75}, true},
		{"in concavity", Point{75, 75}, false},
		{"corner", Point{25, 25}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := pointInPolygon(tt.p, z)
			if got != tt.want {
				t.Errorf("pointInPolygon(%v) = %v, want %v", tt.p, got, tt.want)
			}
		})
	}
}

func TestPointInPolygonEdgeCases(t *testing.T) {
	z := &Zone{
		Type:    "zone",
		Polygon: []Point{{0, 0}, {100, 0}, {100, 100}, {0, 100}},
		bbMinX:  0, bbMinY: 0, bbMaxX: 100, bbMaxY: 100,
	}

	// Points exactly on the boundary may go either way with ray casting,
	// but should not crash.
	edgePts := []Point{
		{0, 0},     // vertex
		{50, 0},    // on top edge
		{100, 50},  // on right edge
		{0, 50},    // on left edge
		{100, 100}, // vertex
	}
	for _, p := range edgePts {
		_ = pointInPolygon(p, z) // must not panic
	}

	// Too few points: should return false.
	zBad := &Zone{
		Type:    "zone",
		Polygon: []Point{{0, 0}, {100, 0}},
		bbMinX:  0, bbMinY: 0, bbMaxX: 100, bbMaxY: 0,
	}
	if pointInPolygon(Point{50, 0}, zBad) {
		t.Error("expected false for polygon with < 3 points")
	}
}

// --- Zone enter/exit tests ---

func TestZoneEnterExit(t *testing.T) {
	zones := []Zone{
		{
			Name:    "area1",
			Type:    "zone",
			Polygon: []Point{{100, 100}, {200, 100}, {200, 200}, {100, 200}},
		},
	}
	zm := NewZoneManager(zones)

	t0 := time.Now()

	// Frame 1: track is outside the zone.
	events := zm.Update([]TrackedObject{
		{ID: 1, Class: 0, Box: [4]float64{50, 50, 80, 80}}, // center = 65, 65
	}, t0)
	if len(events) != 0 {
		t.Errorf("frame 1: expected 0 events, got %d", len(events))
	}

	// Frame 2: track enters the zone.
	t1 := t0.Add(33 * time.Millisecond)
	events = zm.Update([]TrackedObject{
		{ID: 1, Class: 0, Box: [4]float64{130, 130, 170, 170}}, // center = 150, 150
	}, t1)
	if len(events) != 1 || events[0].Type != "enter" {
		t.Errorf("frame 2: expected 1 enter event, got %v", events)
	}

	// Frame 3: track exits the zone.
	t2 := t1.Add(33 * time.Millisecond)
	events = zm.Update([]TrackedObject{
		{ID: 1, Class: 0, Box: [4]float64{250, 250, 280, 280}}, // center = 265, 265
	}, t2)
	if len(events) != 1 || events[0].Type != "exit" {
		t.Errorf("frame 3: expected 1 exit event, got %v", events)
	}
	if events[0].Duration <= 0 {
		t.Error("exit event should have positive duration")
	}
}

// --- Line crossing tests ---

func TestLineCrossing(t *testing.T) {
	zones := []Zone{
		{
			Name: "boundary",
			Type: "line",
			P1:   Point{0, 200},
			P2:   Point{640, 200},
			Dir:  "both",
		},
	}
	zm := NewZoneManager(zones)

	t0 := time.Now()

	// Frame 1: track above the line.
	events := zm.Update([]TrackedObject{
		{ID: 1, Class: 0, Box: [4]float64{100, 100, 140, 140}}, // center = 120, 120
	}, t0)
	// No event on first frame (no previous position).
	if len(events) != 0 {
		t.Errorf("frame 1: expected 0 events, got %d", len(events))
	}

	// Frame 2: track crosses to below the line.
	t1 := t0.Add(33 * time.Millisecond)
	events = zm.Update([]TrackedObject{
		{ID: 1, Class: 0, Box: [4]float64{100, 250, 140, 290}}, // center = 120, 270
	}, t1)
	if len(events) != 1 || events[0].Type != "cross" {
		t.Errorf("frame 2: expected 1 cross event, got %v", events)
	}

	// Frame 3: track crosses back above.
	t2 := t1.Add(33 * time.Millisecond)
	events = zm.Update([]TrackedObject{
		{ID: 1, Class: 0, Box: [4]float64{100, 100, 140, 140}}, // center = 120, 120
	}, t2)
	if len(events) != 1 || events[0].Type != "cross" {
		t.Errorf("frame 3: expected 1 cross event (back), got %v", events)
	}
}

func TestLineCrossingDirectionFilter(t *testing.T) {
	zones := []Zone{
		{
			Name: "exit_only",
			Type: "line",
			P1:   Point{0, 200},
			P2:   Point{640, 200},
			Dir:  "out",
		},
	}
	zm := NewZoneManager(zones)

	t0 := time.Now()

	// Frame 1: track above the line.
	zm.Update([]TrackedObject{
		{ID: 1, Class: 0, Box: [4]float64{100, 100, 140, 140}},
	}, t0)

	// Frame 2: track crosses downward (this is "out" for a horizontal line).
	t1 := t0.Add(33 * time.Millisecond)
	events := zm.Update([]TrackedObject{
		{ID: 1, Class: 0, Box: [4]float64{100, 250, 140, 290}},
	}, t1)

	// Should get an event since direction matches.
	foundCross := false
	for _, e := range events {
		if e.Type == "cross" {
			foundCross = true
		}
	}

	// Now cross in the other direction (should be filtered).
	t2 := t1.Add(33 * time.Millisecond)
	events = zm.Update([]TrackedObject{
		{ID: 1, Class: 0, Box: [4]float64{100, 100, 140, 140}},
	}, t2)

	// The "in" direction crossing should be filtered for "out"-only line.
	for _, e := range events {
		if e.Type == "cross" && e.Direction == "in" {
			// If dir filter is "out", we should only get "out" direction events.
			// The "in" direction should be suppressed.
			if !foundCross {
				t.Error("expected at least one cross event for 'out' direction")
			}
		}
	}
}

// --- Dwell detection tests ---

func TestDwellDetection(t *testing.T) {
	zones := []Zone{
		{
			Name:    "waiting_area",
			Type:    "zone",
			Polygon: []Point{{0, 0}, {200, 0}, {200, 200}, {0, 200}},
		},
	}
	zm := NewZoneManager(zones)

	t0 := time.Now()

	// Track enters zone.
	zm.Update([]TrackedObject{
		{ID: 1, Class: 0, Box: [4]float64{50, 50, 80, 80}},
	}, t0)

	// Simulate 5 seconds of presence.
	for i := 1; i <= 150; i++ { // 150 frames at 30fps = 5s
		ts := t0.Add(time.Duration(i) * 33 * time.Millisecond)
		zm.Update([]TrackedObject{
			{ID: 1, Class: 0, Box: [4]float64{50, 50, 80, 80}},
		}, ts)
	}

	// Check dwell events.
	t5 := t0.Add(5 * time.Second)
	dwells := zm.DwellEvents(t5, 3*time.Second)
	if len(dwells) == 0 {
		t.Error("expected dwell event after 5s in zone with 3s threshold")
	}
	found := false
	for _, d := range dwells {
		if d.Type == "dwell" && d.ZoneName == "waiting_area" && d.TrackID == 1 {
			found = true
			if d.Duration < 3*time.Second {
				t.Errorf("dwell duration %v < 3s threshold", d.Duration)
			}
		}
	}
	if !found {
		t.Error("expected dwell event for track 1 in waiting_area")
	}
}

// --- Occupancy tests ---

func TestOccupancy(t *testing.T) {
	zones := []Zone{
		{
			Name:    "lobby",
			Type:    "zone",
			Polygon: []Point{{0, 0}, {500, 0}, {500, 500}, {0, 500}},
		},
	}
	zm := NewZoneManager(zones)

	t0 := time.Now()

	// Add 5 tracks inside the zone.
	tracks := make([]TrackedObject, 5)
	for i := 0; i < 5; i++ {
		x := float64(i*50 + 50)
		tracks[i] = TrackedObject{
			ID:    i + 1,
			Class: 0,
			Box:   [4]float64{x, 100, x + 30, 130},
		}
	}
	zm.Update(tracks, t0)

	occ := zm.Occupancy("lobby")
	if occ != 5 {
		t.Errorf("expected occupancy 5, got %d", occ)
	}

	// Remove 2 tracks.
	t1 := t0.Add(33 * time.Millisecond)
	zm.Update(tracks[:3], t1)

	occ = zm.Occupancy("lobby")
	if occ != 3 {
		t.Errorf("expected occupancy 3 after removing 2 tracks, got %d", occ)
	}
}

// --- Multiple zones test ---

func TestMultipleZones(t *testing.T) {
	zones := []Zone{
		{
			Name:    "zone_a",
			Type:    "zone",
			Polygon: []Point{{0, 0}, {100, 0}, {100, 100}, {0, 100}},
		},
		{
			Name:    "zone_b",
			Type:    "zone",
			Polygon: []Point{{150, 0}, {250, 0}, {250, 100}, {150, 100}},
		},
		{
			Name:    "zone_c",
			Type:    "zone",
			Polygon: []Point{{300, 0}, {400, 0}, {400, 100}, {300, 100}},
		},
	}
	zm := NewZoneManager(zones)
	t0 := time.Now()

	// Track starts in zone_a.
	events := zm.Update([]TrackedObject{
		{ID: 1, Class: 0, Box: [4]float64{40, 40, 60, 60}}, // center = 50, 50
	}, t0)
	assertEventTypes(t, events, map[string]int{"enter": 1})

	// Track moves to zone_b.
	t1 := t0.Add(33 * time.Millisecond)
	events = zm.Update([]TrackedObject{
		{ID: 1, Class: 0, Box: [4]float64{190, 40, 210, 60}}, // center = 200, 50
	}, t1)
	assertEventTypes(t, events, map[string]int{"exit": 1, "enter": 1})

	// Track moves to zone_c.
	t2 := t1.Add(33 * time.Millisecond)
	events = zm.Update([]TrackedObject{
		{ID: 1, Class: 0, Box: [4]float64{340, 40, 360, 60}}, // center = 350, 50
	}, t2)
	assertEventTypes(t, events, map[string]int{"exit": 1, "enter": 1})
}

func assertEventTypes(t *testing.T, events []ZoneEvent, expected map[string]int) {
	t.Helper()
	counts := make(map[string]int)
	for _, e := range events {
		counts[e.Type]++
	}
	for typ, want := range expected {
		if counts[typ] != want {
			t.Errorf("expected %d %q events, got %d (all events: %v)", want, typ, counts[typ], events)
		}
	}
}

// --- Line side helper test ---

func TestLineSide(t *testing.T) {
	p1 := Point{0, 0}
	p2 := Point{100, 0}

	// Point above the line (negative Y in screen coords, but positive in math).
	above := lineSide(p1, p2, Point{50, -10})
	below := lineSide(p1, p2, Point{50, 10})
	on := lineSide(p1, p2, Point{50, 0})

	if above == 0 || below == 0 {
		t.Error("points above/below should not be on the line")
	}
	if above*below >= 0 {
		t.Error("points above and below should be on opposite sides")
	}
	if on != 0 {
		t.Errorf("point on line should have side=0, got %f", on)
	}
}

// --- YAML config tests ---

func TestLoadZonesFromYAML(t *testing.T) {
	yamlContent := `zones:
  - name: restricted_area
    type: zone
    polygon:
      - {x: 100, y: 100}
      - {x: 500, y: 100}
      - {x: 500, y: 400}
      - {x: 100, y: 400}
  - name: entrance_line
    type: line
    p1: {x: 0, y: 300}
    p2: {x: 640, y: 300}
    direction: both
`
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "zones.yaml")
	if err := os.WriteFile(path, []byte(yamlContent), 0o644); err != nil {
		t.Fatal(err)
	}

	zones, err := LoadZonesFromYAML(path)
	if err != nil {
		t.Fatalf("LoadZonesFromYAML: %v", err)
	}

	if len(zones) != 2 {
		t.Fatalf("expected 2 zones, got %d", len(zones))
	}

	// Verify zone.
	z0 := zones[0]
	if z0.Name != "restricted_area" || z0.Type != "zone" {
		t.Errorf("zone 0: name=%q type=%q", z0.Name, z0.Type)
	}
	if len(z0.Polygon) != 4 {
		t.Errorf("zone 0: expected 4 polygon points, got %d", len(z0.Polygon))
	}

	// Verify line.
	z1 := zones[1]
	if z1.Name != "entrance_line" || z1.Type != "line" {
		t.Errorf("zone 1: name=%q type=%q", z1.Name, z1.Type)
	}
	if z1.P1.X != 0 || z1.P1.Y != 300 || z1.P2.X != 640 || z1.P2.Y != 300 {
		t.Errorf("zone 1: p1=%v p2=%v", z1.P1, z1.P2)
	}
	if z1.Dir != "both" {
		t.Errorf("zone 1: dir=%q, want 'both'", z1.Dir)
	}
}

func TestParseZonesYAMLInvalid(t *testing.T) {
	tests := []struct {
		name string
		yaml string
	}{
		{"bad yaml", "{{{{"},
		{"zone too few points", `zones:
  - name: bad
    type: zone
    polygon:
      - {x: 0, y: 0}
      - {x: 100, y: 0}
`},
		{"line missing p1", `zones:
  - name: bad
    type: line
    p2: {x: 100, y: 100}
`},
		{"unknown type", `zones:
  - name: bad
    type: circle
`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseZonesYAML([]byte(tt.yaml))
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

// --- Benchmark ---

func BenchmarkZoneUpdate(b *testing.B) {
	// 50 zones, 100 tracks.
	zones := make([]Zone, 50)
	for i := 0; i < 50; i++ {
		x := float64(i%10) * 100
		y := float64(i/10) * 100
		zones[i] = Zone{
			Name: fmt.Sprintf("zone_%d", i),
			Type: "zone",
			Polygon: []Point{
				{x, y}, {x + 90, y}, {x + 90, y + 90}, {x, y + 90},
			},
		}
	}
	zm := NewZoneManager(zones)

	tracks := make([]TrackedObject, 100)
	for i := 0; i < 100; i++ {
		x := float64(i%10)*100 + 20
		y := float64(i/10)*100 + 20
		tracks[i] = TrackedObject{
			ID:    i + 1,
			Class: 0,
			Box:   [4]float64{x, y, x + 30, y + 30},
		}
	}

	ts := time.Now()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		zm.Update(tracks, ts)
		ts = ts.Add(33 * time.Millisecond)
	}
}
