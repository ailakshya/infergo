// Package tracker implements ByteTrack multi-object tracking for use with
// YOLO-style object detectors. It associates detections across video frames
// to produce stable track IDs suitable for counting, path analysis, and
// behavior recognition.
//
// Usage:
//
//	bt := tracker.NewByteTracker(tracker.DefaultConfig())
//	for frame := range frames {
//	    dets := detector.Detect(frame)
//	    tracks := bt.Update(dets)
//	    for _, t := range tracks {
//	        fmt.Printf("ID=%d class=%d box=%v\n", t.ID, t.Class, t.Box)
//	    }
//	}
package tracker

// Detection represents a single object detection from a detector (e.g. YOLO).
type Detection struct {
	Box        [4]float64 // x1, y1, x2, y2 (top-left, bottom-right)
	Class      int
	Confidence float64
}

// TrackState represents the lifecycle state of a track.
type TrackState int

const (
	StateNew        TrackState = iota // Just created, unconfirmed
	StateTracked                      // Actively tracked
	StateLost                         // Temporarily lost, may recover
	StateRemoved                      // Expired, will be garbage-collected
)

// Track represents a tracked object with a persistent ID across frames.
type Track struct {
	ID         int
	Class      int
	Confidence float64
	Box        [4]float64 // x1, y1, x2, y2
	State      TrackState
	Age        int // total frames since creation
	TimeLost   int // consecutive frames without a match
}

// strack is the internal representation of a tracked object, holding
// Kalman filter state and association metadata.
type strack struct {
	id         int
	kf         kalmanState
	class      int
	confidence float64
	state      TrackState
	isActive   bool // has been matched at least once
	age        int
	timeLost   int
	startFrame int
}

func (s *strack) box() [4]float64 {
	return xyahToXYXY(s.kf.mean)
}

func (s *strack) toTrack() Track {
	return Track{
		ID:         s.id,
		Class:      s.class,
		Confidence: s.confidence,
		Box:        s.box(),
		State:      s.state,
		Age:        s.age,
		TimeLost:   s.timeLost,
	}
}

// xyxyToXYAH converts [x1,y1,x2,y2] to [cx,cy,aspect_ratio,height].
func xyxyToXYAH(box [4]float64) [4]float64 {
	w := box[2] - box[0]
	h := box[3] - box[1]
	cx := box[0] + w/2
	cy := box[1] + h/2
	if h == 0 {
		h = 1
	}
	return [4]float64{cx, cy, w / h, h}
}

// xyahToXYXY converts [cx,cy,aspect_ratio,height] to [x1,y1,x2,y2].
func xyahToXYXY(mean [8]float64) [4]float64 {
	cx, cy, a, h := mean[0], mean[1], mean[2], mean[3]
	w := a * h
	return [4]float64{
		cx - w/2,
		cy - h/2,
		cx + w/2,
		cy + h/2,
	}
}
