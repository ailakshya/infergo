package main

import (
	"reflect"
	"testing"
)

func TestParseTensorSplit(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    []float32
		wantErr bool
	}{
		{
			name:  "empty string returns nil (single GPU)",
			input: "",
			want:  nil,
		},
		{
			name:  "single GPU full weight",
			input: "1.0",
			want:  []float32{1.0},
		},
		{
			name:  "two GPUs even split",
			input: "0.5,0.5",
			want:  []float32{0.5, 0.5},
		},
		{
			name:  "three GPUs uneven split",
			input: "0.25,0.25,0.5",
			want:  []float32{0.25, 0.25, 0.5},
		},
		{
			name:  "whitespace trimmed",
			input: " 0.5 , 0.5 ",
			want:  []float32{0.5, 0.5},
		},
		{
			name:    "invalid value returns error",
			input:   "0.5,abc",
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseTensorSplit(tc.input)
			if tc.wantErr {
				if err == nil {
					t.Errorf("parseTensorSplit(%q) expected error, got nil", tc.input)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseTensorSplit(%q) unexpected error: %v", tc.input, err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("parseTensorSplit(%q) = %v, want %v", tc.input, got, tc.want)
			}
		})
	}
}
