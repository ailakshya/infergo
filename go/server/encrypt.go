package server

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
)

// deriveKey derives a 32-byte AES-256 key from a passphrase using SHA-256.
func deriveKey(passphrase string) []byte {
	h := sha256.Sum256([]byte(passphrase))
	return h[:]
}

// EncryptFile encrypts the file at src and writes the ciphertext to dst using
// AES-256-GCM. The key is derived from passphrase via SHA-256.
// The output format is: [12-byte nonce][ciphertext+tag].
func EncryptFile(src, dst, passphrase string) error {
	plaintext, err := os.ReadFile(src)
	if err != nil {
		return fmt.Errorf("encrypt: read source: %w", err)
	}

	key := deriveKey(passphrase)
	block, err := aes.NewCipher(key)
	if err != nil {
		return fmt.Errorf("encrypt: new cipher: %w", err)
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return fmt.Errorf("encrypt: new GCM: %w", err)
	}

	nonce := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return fmt.Errorf("encrypt: generate nonce: %w", err)
	}

	ciphertext := gcm.Seal(nonce, nonce, plaintext, nil)

	if err := os.WriteFile(dst, ciphertext, 0600); err != nil {
		return fmt.Errorf("encrypt: write output: %w", err)
	}
	return nil
}

// DecryptReader returns an io.Reader that yields the decrypted plaintext
// from the given encrypted reader. The entire ciphertext is read into memory,
// decrypted via AES-256-GCM, and returned as a bytes.Reader.
// The input format must be: [12-byte nonce][ciphertext+tag].
func DecryptReader(r io.Reader, passphrase string) (io.Reader, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("decrypt: read input: %w", err)
	}

	key := deriveKey(passphrase)
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("decrypt: new cipher: %w", err)
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("decrypt: new GCM: %w", err)
	}

	nonceSize := gcm.NonceSize()
	if len(data) < nonceSize {
		return nil, fmt.Errorf("decrypt: ciphertext too short (need at least %d bytes for nonce)", nonceSize)
	}

	nonce, ciphertext := data[:nonceSize], data[nonceSize:]
	plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("decrypt: authentication failed: %w", err)
	}

	return bytes.NewReader(plaintext), nil
}
