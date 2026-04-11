#!/usr/bin/env python3
"""
validate_export.py — Compare a PyTorch model vs its exported counterpart on random inputs.

Outputs JSON to stdout: {"samples": 100, "max_diff": 0.0003, "passed": true}
Exit code 0 = passed, 1 = failed or error.

Usage:
  # Validate TorchScript export
  python tools/validate_export.py --source models/yolo11n.pt --export models/yolo11n.torchscript.pt

  # Validate ONNX export with custom tolerance
  python tools/validate_export.py --source models/yolo11n.pt --export models/yolo11n.onnx --tolerance 1e-3

  # Quick check with fewer samples
  python tools/validate_export.py --source models/yolo11n.pt --export models/yolo11n.torchscript.pt --samples 10
"""

import argparse
import json
import sys

import torch
import numpy as np


def load_source_model(path):
    """Load the source model using ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("validate: pip install ultralytics")

    model = YOLO(path)
    model.model.eval()
    if hasattr(model.model, "to"):
        model.model.to("cpu")
    return model


def load_export_model(path):
    """Load an exported model (ONNX or TorchScript)."""
    if path.endswith(".onnx"):
        try:
            import onnxruntime as ort
        except ImportError:
            sys.exit("validate: pip install onnxruntime")
        return ("onnx", ort.InferenceSession(path, providers=["CPUExecutionProvider"]))
    else:
        model = torch.jit.load(path, map_location="cpu")
        model.eval()
        return ("torchscript", model)


def run_source(model, inp):
    """Run inference on the source ultralytics model."""
    with torch.no_grad():
        out = model.model(inp)
        if isinstance(out, (tuple, list)):
            out = out[0]
    return out


def run_export(kind, model, inp):
    """Run inference on the exported model."""
    if kind == "onnx":
        # Determine the input name from the ONNX session.
        input_name = model.get_inputs()[0].name
        out = model.run(None, {input_name: inp.numpy()})[0]
        return torch.from_numpy(out)
    else:
        with torch.no_grad():
            out = model(inp)
            if isinstance(out, (tuple, list)):
                out = out[0]
        return out


def main():
    p = argparse.ArgumentParser(
        description="Compare PyTorch model vs exported model on random inputs"
    )
    p.add_argument("--source", required=True, help="original PyTorch model (.pt)")
    p.add_argument("--export", required=True, help="exported model (.onnx or .torchscript.pt)")
    p.add_argument("--samples", type=int, default=100, help="number of random inputs to test")
    p.add_argument("--tolerance", type=float, default=1e-4, help="max allowed output difference")
    args = p.parse_args()

    # Load models.
    print(f"Loading source model: {args.source}", file=sys.stderr)
    src = load_source_model(args.source)

    print(f"Loading export model: {args.export}", file=sys.stderr)
    kind, exp = load_export_model(args.export)

    max_diff = 0.0
    for i in range(args.samples):
        inp = torch.randn(1, 3, 640, 640)

        out_src = run_source(src, inp)
        out_exp = run_export(kind, exp, inp)

        # Ensure both are float tensors for comparison.
        out_src = out_src.float()
        out_exp = out_exp.float()

        diff = (out_src - out_exp).abs().max().item()
        max_diff = max(max_diff, diff)

        if (i + 1) % 10 == 0:
            print(f"  sample {i + 1}/{args.samples}, running max_diff={max_diff:.6g}", file=sys.stderr)

    passed = max_diff <= args.tolerance
    result = {"samples": args.samples, "max_diff": max_diff, "passed": passed}

    # JSON result goes to stdout (parsed by the Go caller).
    print(json.dumps(result))

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
