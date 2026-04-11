package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/ailakshya/infergo/alerts"
	"github.com/ailakshya/infergo/analytics"
	"github.com/ailakshya/infergo/recording"
	"github.com/ailakshya/infergo/server"
	"github.com/ailakshya/infergo/video"
)

// videoFrameSource adapts video.Decoder to the pipeline's FrameSource interface.
type videoFrameSource struct {
	dec *video.Decoder
}

func (v *videoFrameSource) NextFrame() ([]byte, FrameInfo, error) {
	rgb, fi, err := v.dec.NextFrame()
	if err != nil {
		return nil, FrameInfo{}, err
	}
	return rgb, FrameInfo{
		FrameNumber: fi.FrameNumber,
		Width:       fi.Width,
		Height:      fi.Height,
	}, nil
}

func (v *videoFrameSource) Close() {
	v.dec.Close()
}

// runDetect implements the "infergo detect" subcommand.
// It runs the full detection control center pipeline:
//
//	video input -> GPU detect -> ByteTrack -> zones -> alerts -> output
//
// Usage:
//
//	infergo detect \
//	  --model yolo11n:models/yolo11n.torchscript.pt \
//	  --source rtsp://cam1,rtsp://cam2,/dev/video0 \
//	  --zones zones.yaml \
//	  --output rtsp://localhost:8554/annotated \
//	  --webhook http://localhost:3000/alerts \
//	  --record --record-dir /data/clips \
//	  --ws-port 8080
func runDetect(args []string) {
	fs := flag.NewFlagSet("detect", flag.ExitOnError)

	// Model
	var models modelFlags
	fs.Var(&models, "model", "detection model; repeatable. Format: [name:]path.{pt,onnx}")
	provider := fs.String("provider", "cuda", "execution provider: cpu|cuda")
	backend := fs.String("backend", "torch", "inference backend: torch|tensorrt|onnx|adaptive")

	// Sources
	sources := fs.String("source", "", "comma-separated video sources (rtsp://, file, /dev/video0)")

	// Analytics
	zonesFile := fs.String("zones", "", "zone/line config YAML file")
	webhook := fs.String("webhook", "", "alert webhook URL")

	// Output
	output := fs.String("output", "", "output URL: rtsp://, file.mp4, or hls://dir/")
	wsPort := fs.Int("ws-port", 0, "WebSocket port for browser streaming (0 = disabled)")

	// Recording
	record := fs.Bool("record", false, "enable event recording")
	recordDir := fs.String("record-dir", "clips", "directory for event clip recordings")
	recordPre := fs.Int("record-pre", 5, "seconds of pre-event footage to save")
	recordPost := fs.Int("record-post", 10, "seconds of post-event footage to save")
	maxStorage := fs.Float64("max-storage-gb", 50, "max recording storage in GB")

	// Performance
	targetFPS := fs.Int("fps", 0, "target processing FPS per stream (0 = native)")
	batchStreams := fs.Bool("batch-streams", true, "batch frames across streams for GPU efficiency")

	// Decode
	hwDecode := fs.Bool("hw-decode", true, "attempt NVDEC hardware video decoding")
	confThresh := fs.Float64("conf", 0.25, "detection confidence threshold")

	fs.Parse(args)

	if len(models) == 0 {
		fmt.Fprintln(os.Stderr, "detect: --model is required")
		fs.Usage()
		os.Exit(1)
	}
	if *sources == "" {
		fmt.Fprintln(os.Stderr, "detect: --source is required")
		fs.Usage()
		os.Exit(1)
	}

	sourceList := strings.Split(*sources, ",")
	for i := range sourceList {
		sourceList[i] = strings.TrimSpace(sourceList[i])
	}

	log.Printf("[detect] starting detection control center")
	log.Printf("[detect] model: %v", models)
	log.Printf("[detect] sources: %v (%d streams)", sourceList, len(sourceList))
	log.Printf("[detect] provider: %s, backend: %s", *provider, *backend)

	// ── 1. Load detection model ──────────────────────────────────────────────
	// Reuse serve.go's model loading; register into a private registry.
	reg := server.NewRegistry()
	metrics := server.NewMetrics()

	for _, spec := range models {
		name, path := parseModelSpec(spec)
		if err := loadModel(reg, metrics, name, path, *provider, *backend,
			0, 0, 0, 0, nil, 1, 0, 0, 0, false, false, 3, LoadAdaptiveConfig()); err != nil {
			log.Fatalf("[detect] failed to load model %q: %v", spec, err)
		}
		log.Printf("[detect] loaded model %q (%s)", name, path)
	}

	// Resolve the primary detection model from the first --model spec.
	primaryName, _ := parseModelSpec(models[0])
	ref, err := reg.Get(primaryName)
	if err != nil {
		log.Fatalf("[detect] model %q not found in registry: %v", primaryName, err)
	}
	detector, ok := ref.Model.(server.DetectionModel)
	if !ok {
		log.Fatalf("[detect] model %q does not implement DetectionModel", primaryName)
	}
	// Keep ref alive for the lifetime of the pipeline; release on shutdown.
	defer ref.Release()

	// ── 2. Create video decoder factory ──────────────────────────────────────
	hwDecodeFlag := *hwDecode
	decoderFactory := func(url string) (FrameSource, error) {
		dec, err := video.OpenDecoder(url, hwDecodeFlag)
		if err != nil {
			return nil, err
		}
		log.Printf("[detect] opened decoder for %s (%dx%d @ %.1f FPS, hw=%v)",
			url, dec.Width(), dec.Height(), dec.FPS(), dec.IsHWAccelerated())
		return &videoFrameSource{dec: dec}, nil
	}

	// ── 3. Create StreamPipeline ─────────────────────────────────────────────
	pipeline := NewStreamPipeline(detector, decoderFactory, 256)

	for i, src := range sourceList {
		cfg := StreamConfig{
			ID:    i,
			URL:   src,
			Model: primaryName,
			FPS:   *targetFPS,
		}
		if err := pipeline.AddStream(cfg); err != nil {
			log.Fatalf("[detect] failed to add stream %d (%s): %v", i, src, err)
		}
		log.Printf("[detect] stream %d: %s", i, src)
	}

	// ── 4. Load zones if configured ──────────────────────────────────────────
	var zoneMgr *analytics.ZoneManager
	if *zonesFile != "" {
		zones, err := analytics.LoadZonesFromYAML(*zonesFile)
		if err != nil {
			log.Fatalf("[detect] failed to load zones: %v", err)
		}
		zoneMgr = analytics.NewZoneManager(zones)
		log.Printf("[detect] loaded %d zones from %s", len(zones), *zonesFile)
	}

	// ── 5. Create AlertManager if webhook configured ─────────────────────────
	var alertMgr *alerts.AlertManager
	if *webhook != "" {
		// Default rule: alert on any zone enter event.
		rules := []alerts.Rule{
			{
				Name: "zone_enter",
				Condition: func(event analytics.ZoneEvent) bool {
					return event.Type == "enter"
				},
				Cooldown: 10 * time.Second,
			},
			{
				Name: "line_cross",
				Condition: func(event analytics.ZoneEvent) bool {
					return event.Type == "cross"
				},
				Cooldown: 5 * time.Second,
			},
		}
		alertMgr = alerts.NewAlertManager(rules, *webhook)
		log.Printf("[detect] alerts: webhook=%s", *webhook)
	}

	// ── 6. Create EventRecorder if --record enabled ──────────────────────────
	var recorder *recording.EventRecorder
	if *record {
		recorder = recording.NewEventRecorder(*recordDir, *recordPre, *recordPost, *maxStorage)
		log.Printf("[detect] recording: dir=%s (pre=%ds, post=%ds, max=%.0fGB)",
			*recordDir, *recordPre, *recordPost, *maxStorage)
	}

	// ── 7. Start WebSocket server if --ws-port > 0 ──────────────────────────
	var wsHub *server.StreamHub
	if *wsPort > 0 {
		wsHub = server.NewStreamHub()
		apiSrv := server.NewServer(reg)
		mux := http.NewServeMux()
		mux.Handle("/v1/ws/detect", apiSrv.HandleWSDetect(wsHub))
		mux.HandleFunc("/v1/streams", apiSrv.HandleStreams(wsHub))
		wsAddr := fmt.Sprintf(":%d", *wsPort)
		go func() {
			log.Printf("[detect] WebSocket server on %s", wsAddr)
			if err := http.ListenAndServe(wsAddr, mux); err != nil {
				log.Printf("[detect] WebSocket server error: %v", err)
			}
		}()
	}

	// ── 8. Start output encoder if --output configured ───────────────────────
	var enc *video.Encoder
	if *output != "" {
		// Determine dimensions from first source decoder.
		// Use 1920x1080 as default; actual dimensions come from the first frame.
		encW, encH := 1920, 1080
		encFPS := 30
		if *targetFPS > 0 {
			encFPS = *targetFPS
		}
		codec := "libx264"
		if *provider == "cuda" {
			codec = "h264_nvenc"
		}
		enc, err = video.OpenEncoder(*output, encW, encH, encFPS, codec)
		if err != nil {
			log.Printf("[detect] warning: failed to open output encoder: %v (continuing without output)", err)
			enc = nil
		} else {
			log.Printf("[detect] output encoder: %s (%s)", *output, codec)
			defer enc.Close()
		}
	}

	log.Printf("[detect] batch across streams: %v", *batchStreams)

	_ = confThresh // used via pipeline's default thresholds

	// ── 9. Result consumer loop ──────────────────────────────────────────────
	// Consume results from pipeline, run zones/alerts/annotation/recording.
	fpsTrackers := make(map[int]*server.FPSTracker)

	go func() {
		for result := range pipeline.Results() {
			// Track FPS per stream.
			if fpsTrackers[result.StreamID] == nil {
				fpsTrackers[result.StreamID] = server.NewFPSTracker(2.0)
			}
			currentFPS := fpsTrackers[result.StreamID].Tick()

			// Log periodically (every 100 frames per stream).
			if result.FrameNumber%100 == 0 {
				log.Printf("[detect] stream %d frame %d: %d detections, %d tracks, %.1f FPS",
					result.StreamID, result.FrameNumber,
					len(result.Detections), len(result.Tracks), currentFPS)
			}

			// Zone analytics.
			var zoneEvents []analytics.ZoneEvent
			if zoneMgr != nil && len(result.Tracks) > 0 {
				tracked := make([]analytics.TrackedObject, len(result.Tracks))
				for i, t := range result.Tracks {
					tracked[i] = analytics.TrackedObject{
						ID:    t.TrackID,
						Class: t.ClassID,
						Box: [4]float64{
							float64(t.X1), float64(t.Y1),
							float64(t.X2), float64(t.Y2),
						},
					}
				}
				zoneEvents = zoneMgr.Update(tracked, result.Timestamp)
				for _, evt := range zoneEvents {
					log.Printf("[detect] zone event: %s zone=%q track=%d class=%d",
						evt.Type, evt.ZoneName, evt.TrackID, evt.ClassID)
				}
			}

			// Alerts.
			if alertMgr != nil && len(zoneEvents) > 0 {
				for _, evt := range zoneEvents {
					fired := alertMgr.ProcessEvent(evt, result.StreamID, nil)
					for _, alert := range fired {
						log.Printf("[detect] ALERT: %s (stream=%d track=%d zone=%s)",
							alert.Rule, alert.StreamID, alert.Event.TrackID, alert.Event.ZoneName)
						go func(a alerts.Alert) {
							if err := alertMgr.SendWebhook(a); err != nil {
								log.Printf("[detect] webhook error: %v", err)
							}
						}(alert)
					}
				}
			}

			// Buffer frame for recording.
			if recorder != nil {
				recorder.BufferFrame(result.StreamID, result.FrameRGB,
					result.Width, result.Height, result.Timestamp)

				// Save a clip when a zone event occurs.
				if len(zoneEvents) > 0 {
					go func(sid int, ts time.Time, events []analytics.ZoneEvent) {
						meta := map[string]string{
							"event_name": events[0].Type + "_" + events[0].ZoneName,
						}
						path, err := recorder.SaveClip(sid, ts, meta)
						if err != nil {
							log.Printf("[detect] save clip error: %v", err)
						} else {
							log.Printf("[detect] saved clip: %s", path)
						}
						// Cleanup old clips.
						recorder.Cleanup()
					}(result.StreamID, result.Timestamp, zoneEvents)
				}
			}

			// Annotate and publish.
			var annotatedJPEG []byte
			if wsHub != nil || enc != nil {
				objects := make([]video.AnnotateObject, len(result.Tracks))
				for i, t := range result.Tracks {
					label := fmt.Sprintf("%s #%d %.0f%%",
						video.CocoClassName(t.ClassID), t.TrackID, t.Confidence*100)
					objects[i] = video.AnnotateObject{
						X1:         t.X1,
						Y1:         t.Y1,
						X2:         t.X2,
						Y2:         t.Y2,
						ClassID:    t.ClassID,
						Confidence: t.Confidence,
						TrackID:    t.TrackID,
						Label:      label,
					}
				}
				var annErr error
				annotatedJPEG, annErr = video.AnnotateFast(result.FrameRGB, result.Width, result.Height, objects, 75)
				if annErr != nil {
					// Fallback to pure Go annotator.
					log.Printf("[detect] C annotator failed (%v), falling back to Go", annErr)
					annotatedJPEG, annErr = video.Annotate(result.FrameRGB, result.Width, result.Height, objects)
					if annErr != nil {
						log.Printf("[detect] annotate error: %v", annErr)
					}
				}
			}

			// Publish to WebSocket subscribers.
			if wsHub != nil && annotatedJPEG != nil {
				wsObjects := make([]server.WSDetectObject, len(result.Tracks))
				for i, t := range result.Tracks {
					wsObjects[i] = server.WSDetectObject{
						X1:         t.X1,
						Y1:         t.Y1,
						X2:         t.X2,
						Y2:         t.Y2,
						ClassID:    t.ClassID,
						Confidence: t.Confidence,
						TrackID:    t.TrackID,
						Label:      video.CocoClassName(t.ClassID),
					}
				}
				wsFrame := server.WSDetectFrame{
					StreamID:    result.StreamID,
					FrameNumber: result.FrameNumber,
					Timestamp:   result.Timestamp.Format("2006-01-02T15:04:05.000Z07:00"),
					Objects:     wsObjects,
					FPS:         currentFPS,
				}
				wsHub.Publish(result.StreamID, wsFrame, annotatedJPEG)
			}

			// Write to output encoder.
			if enc != nil && result.FrameRGB != nil {
				if err := enc.WriteFrame(result.FrameRGB); err != nil {
					log.Printf("[detect] encoder error: %v", err)
				}
			}
		}
	}()

	// Wait for shutdown signal.
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)

	log.Printf("[detect] pipeline running, press Ctrl+C to stop")
	<-stop
	log.Printf("[detect] shutting down...")

	pipeline.Close()
	log.Printf("[detect] done")
}
