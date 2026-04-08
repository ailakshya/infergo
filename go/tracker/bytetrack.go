package tracker

// Config holds ByteTrack tracker parameters.
// These match the fields in bytetrack YAML config files used by ultralytics.
type Config struct {
	TrackHighThresh float64 // High confidence threshold for primary association (default 0.20)
	TrackLowThresh  float64 // Low confidence threshold for secondary association (default 0.08)
	NewTrackThresh  float64 // Minimum confidence to initialize a new track (default 0.25)
	TrackBuffer     int     // Max frames to keep a lost track alive (default 60)
	MatchThresh     float64 // IoU threshold for matching (default 0.80)
	FuseScore       bool    // Fuse detection confidence with IoU distance (default true)
}

// DefaultConfig returns a Config with sensible defaults matching
// the standard ByteTrack configuration.
func DefaultConfig() Config {
	return Config{
		TrackHighThresh: 0.20,
		TrackLowThresh:  0.08,
		NewTrackThresh:  0.25,
		TrackBuffer:     60,
		MatchThresh:     0.80,
		FuseScore:       true,
	}
}

// ByteTracker implements the ByteTrack multi-object tracking algorithm.
// It maintains persistent track IDs across frames using a 3-stage
// association strategy that leverages both high and low confidence detections.
//
// Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
// Every Detection Box", ECCV 2022.
type ByteTracker struct {
	cfg        Config
	nextID     int
	frameCount int

	tracked     []*strack // actively tracked
	lost        []*strack // temporarily lost
	removed     []*strack // expired (kept briefly for dedup)
	unconfirmed []*strack // newly created, not yet matched twice
}

// NewByteTracker creates a new ByteTracker with the given configuration.
func NewByteTracker(cfg Config) *ByteTracker {
	return &ByteTracker{cfg: cfg, nextID: 1}
}

// Update processes a new frame's detections and returns the currently active tracks.
// Call this once per frame with all detections from the detector.
func (bt *ByteTracker) Update(detections []Detection) []Track {
	bt.frameCount++

	// Split detections into high and low confidence
	var highDets, lowDets []Detection
	for _, d := range detections {
		if d.Confidence >= bt.cfg.TrackHighThresh {
			highDets = append(highDets, d)
		} else if d.Confidence >= bt.cfg.TrackLowThresh {
			lowDets = append(lowDets, d)
		}
	}

	// Separate tracked vs unconfirmed from previous frame
	var trackedStracks, unconfirmedStracks []*strack
	for _, t := range bt.tracked {
		if t.isActive {
			trackedStracks = append(trackedStracks, t)
		} else {
			unconfirmedStracks = append(unconfirmedStracks, t)
		}
	}

	// Merge tracked + lost for first association pool
	pool := append(trackedStracks, bt.lost...)

	// Predict all tracks forward
	for _, t := range pool {
		t.kf.predict()
	}
	for _, t := range unconfirmedStracks {
		t.kf.predict()
	}

	// ── Stage 1: Associate high-confidence detections with all tracks ────

	var matchedRows, matchedCols []int
	var unmatchedTracks, unmatchedDets []int

	if len(pool) > 0 && len(highDets) > 0 {
		dist := iouDistance(pool, highDets)
		if bt.cfg.FuseScore {
			dist = fuseScore(dist, highDets)
		}
		matchedRows, matchedCols, unmatchedTracks, unmatchedDets =
			linearAssignment(dist, bt.cfg.MatchThresh)
	} else {
		for i := range pool {
			unmatchedTracks = append(unmatchedTracks, i)
		}
		for i := range highDets {
			unmatchedDets = append(unmatchedDets, i)
		}
	}

	// Apply matches
	var activatedTracks []*strack
	var refoundTracks []*strack

	for i := range matchedRows {
		track := pool[matchedRows[i]]
		det := highDets[matchedCols[i]]
		bt.updateTrack(track, det)
		if track.state == StateLost {
			refoundTracks = append(refoundTracks, track)
		} else {
			activatedTracks = append(activatedTracks, track)
		}
	}

	// ── Stage 2: Associate low-confidence detections with remaining tracked ──

	var remainTracked []*strack
	for _, idx := range unmatchedTracks {
		t := pool[idx]
		if t.state == StateTracked {
			remainTracked = append(remainTracked, t)
		}
	}

	var matchedRows2, matchedCols2 []int
	var unmatchedTracks2 []int

	if len(remainTracked) > 0 && len(lowDets) > 0 {
		dist2 := iouDistance(remainTracked, lowDets)
		matchedRows2, matchedCols2, unmatchedTracks2, _ =
			linearAssignment(dist2, 0.5)
	} else {
		for i := range remainTracked {
			unmatchedTracks2 = append(unmatchedTracks2, i)
		}
	}

	for i := range matchedRows2 {
		track := remainTracked[matchedRows2[i]]
		det := lowDets[matchedCols2[i]]
		bt.updateTrack(track, det)
		activatedTracks = append(activatedTracks, track)
	}

	// Mark unmatched tracked objects as lost
	var newLost []*strack
	for _, idx := range unmatchedTracks2 {
		t := remainTracked[idx]
		if t.state != StateLost {
			t.state = StateLost
			newLost = append(newLost, t)
		}
	}
	// Also mark lost tracks from stage 1 that weren't recovered
	for _, idx := range unmatchedTracks {
		t := pool[idx]
		if t.state == StateLost {
			t.timeLost++
			if t.timeLost > bt.cfg.TrackBuffer {
				t.state = StateRemoved
			}
		}
	}

	// ── Stage 3: Associate unconfirmed tracks with remaining high-conf dets ──

	var remainHighDets []Detection
	for _, idx := range unmatchedDets {
		remainHighDets = append(remainHighDets, highDets[idx])
	}

	var matchedRows3, matchedCols3 []int
	var unmatchedUnconf, unmatchedDets3 []int

	if len(unconfirmedStracks) > 0 && len(remainHighDets) > 0 {
		dist3 := iouDistance(unconfirmedStracks, remainHighDets)
		matchedRows3, matchedCols3, unmatchedUnconf, unmatchedDets3 =
			linearAssignment(dist3, 0.7)
	} else {
		for i := range unconfirmedStracks {
			unmatchedUnconf = append(unmatchedUnconf, i)
		}
		for i := range remainHighDets {
			unmatchedDets3 = append(unmatchedDets3, i)
		}
	}

	for i := range matchedRows3 {
		track := unconfirmedStracks[matchedRows3[i]]
		det := remainHighDets[matchedCols3[i]]
		bt.updateTrack(track, det)
		track.isActive = true
		activatedTracks = append(activatedTracks, track)
	}

	// Remove unmatched unconfirmed tracks
	for _, idx := range unmatchedUnconf {
		unconfirmedStracks[idx].state = StateRemoved
	}

	// ── Stage 4: Initialize new tracks from remaining detections ────────

	var newTracks []*strack
	for _, idx := range unmatchedDets3 {
		det := remainHighDets[idx]
		if det.Confidence >= bt.cfg.NewTrackThresh {
			t := bt.newTrack(det)
			newTracks = append(newTracks, t)
		}
	}

	// ── Assemble final track lists ──────────────────────────────────────

	// Update tracked list
	var newTracked []*strack
	for _, t := range activatedTracks {
		t.state = StateTracked
		newTracked = append(newTracked, t)
	}
	for _, t := range refoundTracks {
		t.state = StateTracked
		t.timeLost = 0
		newTracked = append(newTracked, t)
	}
	for _, t := range newTracks {
		newTracked = append(newTracked, t)
	}

	// Update lost list
	var newLostList []*strack
	for _, t := range bt.lost {
		if t.state == StateLost {
			newLostList = append(newLostList, t)
		}
	}
	for _, t := range newLost {
		newLostList = append(newLostList, t)
	}

	// Remove duplicates between tracked and lost
	newTracked, newLostList = removeDuplicates(newTracked, newLostList)

	bt.tracked = newTracked
	bt.lost = newLostList

	// Build output
	var output []Track
	for _, t := range bt.tracked {
		if t.isActive {
			t.age++
			output = append(output, t.toTrack())
		}
	}
	return output
}

// Reset clears all tracker state.
func (bt *ByteTracker) Reset() {
	bt.tracked = nil
	bt.lost = nil
	bt.removed = nil
	bt.unconfirmed = nil
	bt.frameCount = 0
}

// newTrack creates a new strack from a detection.
func (bt *ByteTracker) newTrack(det Detection) *strack {
	xyah := xyxyToXYAH(det.Box)
	id := bt.nextID
	bt.nextID++
	return &strack{
		id:         id,
		kf:         initKalman(xyah),
		class:      det.Class,
		confidence: det.Confidence,
		state:      StateNew,
		isActive:   false,
		age:        1,
		startFrame: bt.frameCount,
	}
}

// updateTrack updates a track with a new matched detection.
func (bt *ByteTracker) updateTrack(t *strack, det Detection) {
	xyah := xyxyToXYAH(det.Box)
	t.kf.update(xyah)
	t.confidence = det.Confidence
	t.class = det.Class
	t.isActive = true
	t.timeLost = 0
}

// removeDuplicates removes tracks that appear in both tracked and lost lists
// based on IoU overlap. Keeps the one with longer tracking history.
func removeDuplicates(tracked, lost []*strack) ([]*strack, []*strack) {
	if len(tracked) == 0 || len(lost) == 0 {
		return tracked, lost
	}

	// Find pairs with high IoU overlap
	keepTracked := make([]bool, len(tracked))
	keepLost := make([]bool, len(lost))
	for i := range keepTracked {
		keepTracked[i] = true
	}
	for i := range keepLost {
		keepLost[i] = true
	}

	for i, t := range tracked {
		for j, l := range lost {
			if iou(t.box(), l.box()) > 0.15 {
				// Keep the one with more history
				if t.age >= l.age {
					keepLost[j] = false
				} else {
					keepTracked[i] = false
				}
			}
		}
	}

	var newTracked []*strack
	for i, t := range tracked {
		if keepTracked[i] {
			newTracked = append(newTracked, t)
		}
	}
	var newLost []*strack
	for i, l := range lost {
		if keepLost[i] {
			newLost = append(newLost, l)
		}
	}
	return newTracked, newLost
}
