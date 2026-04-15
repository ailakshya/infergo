package server

import "testing"

func TestRBACAdmin(t *testing.T) {
	rc := NewRBACConfig(map[string]string{"admin-key": "admin"})
	if !rc.CheckAccess("admin-key", "/v1/admin/reload") {
		t.Fatal("admin should access admin endpoints")
	}
	if !rc.CheckAccess("admin-key", "/v1/chat/completions") {
		t.Fatal("admin should access inference")
	}
}

func TestRBACUser(t *testing.T) {
	rc := NewRBACConfig(map[string]string{"user-key": "user"})
	if !rc.CheckAccess("user-key", "/v1/chat/completions") {
		t.Fatal("user should access inference")
	}
	if rc.CheckAccess("user-key", "/v1/admin/reload") {
		t.Fatal("user should NOT access admin")
	}
}

func TestRBACReadonly(t *testing.T) {
	rc := NewRBACConfig(map[string]string{"ro-key": "readonly"})
	if !rc.CheckAccess("ro-key", "/v1/models") {
		t.Fatal("readonly should access models")
	}
	if rc.CheckAccess("ro-key", "/v1/chat/completions") {
		t.Fatal("readonly should NOT access inference")
	}
	if rc.CheckAccess("ro-key", "/v1/admin/reload") {
		t.Fatal("readonly should NOT access admin")
	}
}

func TestRBACUnknownKey(t *testing.T) {
	rc := NewRBACConfig(map[string]string{"admin-key": "admin"})
	if rc.CheckAccess("unknown-key", "/v1/models") {
		t.Fatal("unknown key should be denied")
	}
}
