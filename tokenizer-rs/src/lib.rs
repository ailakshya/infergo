//! C FFI for HuggingFace tokenizers.
//! Exposes a minimal C API used by infergo's C++ tokenizer wrapper.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;
use tokenizers::Tokenizer;

// ── Opaque handle ─────────────────────────────────────────────────────────────

pub struct TokenizerHandle {
    inner: Tokenizer,
}

// ── Load / free ───────────────────────────────────────────────────────────────

/// Load a tokenizer from a tokenizer.json file.
/// Returns NULL on failure; error written to err_buf (if non-null, buf_len > 0).
#[no_mangle]
pub extern "C" fn tokenizer_load(
    path: *const c_char,
    err_buf: *mut c_char,
    buf_len: c_int,
) -> *mut TokenizerHandle {
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(e) => { write_err(err_buf, buf_len, &e.to_string()); return ptr::null_mut(); }
    };
    match Tokenizer::from_file(path_str) {
        Ok(t) => Box::into_raw(Box::new(TokenizerHandle { inner: t })),
        Err(e) => { write_err(err_buf, buf_len, &e.to_string()); ptr::null_mut() }
    }
}

/// Free a tokenizer handle. Safe to call with NULL.
#[no_mangle]
pub extern "C" fn tokenizer_free(handle: *mut TokenizerHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)); }
    }
}

// ── Encode ────────────────────────────────────────────────────────────────────

/// Encode text into token IDs.
/// Writes up to max_tokens IDs into out_ids and the matching attention mask
/// into out_mask. Returns the actual number of tokens, or -1 on error.
#[no_mangle]
pub extern "C" fn tokenizer_encode(
    handle: *const TokenizerHandle,
    text: *const c_char,
    add_special_tokens: c_int,
    out_ids: *mut c_int,
    out_mask: *mut c_int,
    max_tokens: c_int,
    err_buf: *mut c_char,
    buf_len: c_int,
) -> c_int {
    if handle.is_null() || text.is_null() || out_ids.is_null() { return -1; }
    let t = unsafe { &(*handle).inner };
    let text_str = match unsafe { CStr::from_ptr(text) }.to_str() {
        Ok(s) => s,
        Err(e) => { write_err(err_buf, buf_len, &e.to_string()); return -1; }
    };
    let enc = match t.encode(text_str, add_special_tokens != 0) {
        Ok(e) => e,
        Err(e) => { write_err(err_buf, buf_len, &e.to_string()); return -1; }
    };
    let ids = enc.get_ids();
    let mask = enc.get_attention_mask();
    let n = ids.len().min(max_tokens as usize);
    unsafe {
        for i in 0..n {
            *out_ids.add(i) = ids[i] as c_int;
            if !out_mask.is_null() {
                *out_mask.add(i) = mask[i] as c_int;
            }
        }
    }
    n as c_int
}

// ── Decode ────────────────────────────────────────────────────────────────────

/// Decode token IDs back to text.
/// Writes null-terminated UTF-8 string into out_buf (max buf_len bytes).
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn tokenizer_decode(
    handle: *const TokenizerHandle,
    ids: *const c_int,
    n_ids: c_int,
    skip_special_tokens: c_int,
    out_buf: *mut c_char,
    buf_len: c_int,
    err_buf: *mut c_char,
    err_len: c_int,
) -> c_int {
    if handle.is_null() || ids.is_null() || out_buf.is_null() { return -1; }
    let t = unsafe { &(*handle).inner };
    let id_slice: Vec<u32> = unsafe {
        std::slice::from_raw_parts(ids, n_ids as usize)
            .iter().map(|&x| x as u32).collect()
    };
    match t.decode(&id_slice, skip_special_tokens != 0) {
        Ok(text) => {
            let cs = match CString::new(text) {
                Ok(s) => s,
                Err(e) => { write_err(err_buf, err_len, &e.to_string()); return -1; }
            };
            let bytes = cs.as_bytes_with_nul();
            let copy_n = bytes.len().min(buf_len as usize);
            unsafe {
                ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, out_buf, copy_n);
                // ensure null termination if truncated
                *out_buf.add(buf_len as usize - 1) = 0;
            }
            0
        }
        Err(e) => { write_err(err_buf, err_len, &e.to_string()); -1 }
    }
}

// ── Metadata ─────────────────────────────────────────────────────────────────

/// Returns the vocabulary size, or 0 if handle is null.
#[no_mangle]
pub extern "C" fn tokenizer_vocab_size(handle: *const TokenizerHandle) -> c_int {
    if handle.is_null() { return 0; }
    unsafe { (*handle).inner.get_vocab_size(true) as c_int }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn write_err(buf: *mut c_char, len: c_int, msg: &str) {
    if buf.is_null() || len <= 0 { return; }
    let bytes = msg.as_bytes();
    let n = bytes.len().min(len as usize - 1);
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buf, n);
        *buf.add(n) = 0;
    }
}
