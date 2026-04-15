package server

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestEncryptDecryptRoundTrip(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "plain.txt")
	dst := filepath.Join(dir, "encrypted.bin")
	passphrase := "test-passphrase-123"

	original := []byte("Hello, this is a secret model file with some binary data: \x00\x01\x02\xff")
	if err := os.WriteFile(src, original, 0644); err != nil {
		t.Fatal(err)
	}

	// Encrypt
	if err := EncryptFile(src, dst, passphrase); err != nil {
		t.Fatalf("EncryptFile failed: %v", err)
	}

	// Verify encrypted file exists and differs from original
	enc, err := os.ReadFile(dst)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(enc, original) {
		t.Error("encrypted data should differ from original")
	}

	// Decrypt
	reader, err := DecryptReader(bytes.NewReader(enc), passphrase)
	if err != nil {
		t.Fatalf("DecryptReader failed: %v", err)
	}
	decrypted, err := io.ReadAll(reader)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(decrypted, original) {
		t.Errorf("decrypted data does not match original\ngot:  %q\nwant: %q", decrypted, original)
	}
}

func TestDecryptWrongPassphrase(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "plain.txt")
	dst := filepath.Join(dir, "encrypted.bin")

	if err := os.WriteFile(src, []byte("secret data"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := EncryptFile(src, dst, "correct-passphrase"); err != nil {
		t.Fatal(err)
	}

	enc, _ := os.ReadFile(dst)
	_, err := DecryptReader(bytes.NewReader(enc), "wrong-passphrase")
	if err == nil {
		t.Error("expected error when decrypting with wrong passphrase")
	}
}

func TestDecryptTooShort(t *testing.T) {
	_, err := DecryptReader(bytes.NewReader([]byte("short")), "pass")
	if err == nil {
		t.Error("expected error for ciphertext shorter than nonce")
	}
}

func TestEncryptEmptyFile(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "empty.txt")
	dst := filepath.Join(dir, "encrypted.bin")

	if err := os.WriteFile(src, []byte{}, 0644); err != nil {
		t.Fatal(err)
	}
	if err := EncryptFile(src, dst, "pass"); err != nil {
		t.Fatalf("EncryptFile failed on empty file: %v", err)
	}

	enc, _ := os.ReadFile(dst)
	reader, err := DecryptReader(bytes.NewReader(enc), "pass")
	if err != nil {
		t.Fatalf("DecryptReader failed on empty file: %v", err)
	}
	data, _ := io.ReadAll(reader)
	if len(data) != 0 {
		t.Errorf("expected empty decrypted data, got %d bytes", len(data))
	}
}
