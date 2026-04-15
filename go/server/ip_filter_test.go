package server

import "testing"

func TestIPFilterAllowed(t *testing.T) {
	f := NewIPFilter("10.0.0.0/8,192.168.1.0/24")

	tests := []struct {
		addr    string
		allowed bool
	}{
		{"10.0.0.1:8080", true},
		{"10.255.255.255:80", true},
		{"192.168.1.100:443", true},
		{"192.168.2.1:80", false},
		{"8.8.8.8:53", false},
	}

	for _, tt := range tests {
		if got := f.Allowed(tt.addr); got != tt.allowed {
			t.Errorf("Allowed(%q) = %v, want %v", tt.addr, got, tt.allowed)
		}
	}
}

func TestIPFilterDisabled(t *testing.T) {
	f := NewIPFilter("")
	if !f.Allowed("8.8.8.8:80") {
		t.Fatal("disabled filter should allow all")
	}
}

func TestIPFilterSingleIP(t *testing.T) {
	f := NewIPFilter("127.0.0.1")
	if !f.Allowed("127.0.0.1:9090") {
		t.Fatal("localhost should be allowed")
	}
	if f.Allowed("10.0.0.1:80") {
		t.Fatal("non-localhost should be blocked")
	}
}
