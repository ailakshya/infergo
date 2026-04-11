package analytics

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// zonesConfig is the top-level YAML structure for zone configuration files.
type zonesConfig struct {
	Zones []zoneYAML `yaml:"zones"`
}

// zoneYAML is the YAML representation of a zone.
type zoneYAML struct {
	Name      string      `yaml:"name"`
	Type      string      `yaml:"type"`
	Polygon   []pointYAML `yaml:"polygon,omitempty"`
	P1        *pointYAML  `yaml:"p1,omitempty"`
	P2        *pointYAML  `yaml:"p2,omitempty"`
	Direction string      `yaml:"direction,omitempty"`
}

// pointYAML is the YAML representation of a point.
type pointYAML struct {
	X float64 `yaml:"x"`
	Y float64 `yaml:"y"`
}

// LoadZonesFromYAML reads a YAML configuration file and returns a slice of Zone definitions.
//
// Expected YAML format:
//
//	zones:
//	  - name: restricted_area
//	    type: zone
//	    polygon:
//	      - {x: 100, y: 100}
//	      - {x: 500, y: 100}
//	      - {x: 500, y: 400}
//	      - {x: 100, y: 400}
//	  - name: entrance_line
//	    type: line
//	    p1: {x: 0, y: 300}
//	    p2: {x: 640, y: 300}
//	    direction: both
func LoadZonesFromYAML(path string) ([]Zone, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read zone config %s: %w", path, err)
	}
	return ParseZonesYAML(data)
}

// ParseZonesYAML parses raw YAML bytes into a slice of Zone definitions.
func ParseZonesYAML(data []byte) ([]Zone, error) {
	var cfg zonesConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse zone config: %w", err)
	}

	zones := make([]Zone, 0, len(cfg.Zones))
	for _, zy := range cfg.Zones {
		z := Zone{
			Name: zy.Name,
			Type: zy.Type,
			Dir:  zy.Direction,
		}

		switch zy.Type {
		case "zone":
			if len(zy.Polygon) < 3 {
				return nil, fmt.Errorf("zone %q: polygon requires at least 3 points, got %d", zy.Name, len(zy.Polygon))
			}
			z.Polygon = make([]Point, len(zy.Polygon))
			for i, p := range zy.Polygon {
				z.Polygon[i] = Point{X: p.X, Y: p.Y}
			}
		case "line":
			if zy.P1 == nil || zy.P2 == nil {
				return nil, fmt.Errorf("zone %q: line type requires p1 and p2", zy.Name)
			}
			z.P1 = Point{X: zy.P1.X, Y: zy.P1.Y}
			z.P2 = Point{X: zy.P2.X, Y: zy.P2.Y}
			if z.Dir == "" {
				z.Dir = "both"
			}
		default:
			return nil, fmt.Errorf("zone %q: unknown type %q (expected \"zone\" or \"line\")", zy.Name, zy.Type)
		}

		zones = append(zones, z)
	}
	return zones, nil
}
