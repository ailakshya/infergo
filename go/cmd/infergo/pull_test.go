package main

// HuggingFace hub download tests live in the pure-Go hub package:
//   go test github.com/ailakshya/infergo/hub
//
// This file is intentionally minimal to avoid CGo link failures on machines
// that do not have the C++ inference libraries installed (e.g. the Mac
// development machine).
