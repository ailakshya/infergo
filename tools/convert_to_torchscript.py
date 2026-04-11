#!/usr/bin/env python3
"""
convert_to_torchscript.py — Convert ONNX or ultralytics models to TorchScript (.pt)
or ONNX for use with infergo's torch/onnx backends.

Usage:
  # From ultralytics YOLO model name (downloads + exports to TorchScript)
  python tools/convert_to_torchscript.py --source yolo11n --output models/yolo11n.torchscript.pt

  # From existing .pt ultralytics model
  python tools/convert_to_torchscript.py --source models/yolo11n.pt --output models/yolo11n.torchscript.pt

  # Export to ONNX instead
  python tools/convert_to_torchscript.py --source models/yolo11n.pt --format onnx --output models/yolo11n.onnx

  # From ONNX model to TorchScript (requires onnx2torch)
  python tools/convert_to_torchscript.py --source models/yolo11n.onnx --output models/yolo11n.torchscript.pt

  # Batch convert all YOLO sizes
  python tools/convert_to_torchscript.py --batch yolo11n,yolo11s,yolo11m,yolo11l --output-dir models/
"""

import argparse
import os
import sys
import time

import torch


def convert_ultralytics(source: str, output: str, imgsz: int = 640, fmt: str = "torchscript"):
    """Convert an ultralytics YOLO model to TorchScript or ONNX."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("pip install ultralytics")

    print(f"  Loading ultralytics model: {source}")
    model = YOLO(source)

    if fmt == "onnx":
        print(f"  Exporting to ONNX (imgsz={imgsz})...")
        exported = model.export(format="onnx", imgsz=imgsz, simplify=True)
        print(f"  Exported to: {exported}")

        if exported and os.path.exists(exported) and exported != output:
            os.rename(exported, output)
            print(f"  Moved to: {output}")

        print(f"  Size: {os.path.getsize(output) / 1024 / 1024:.1f} MB")
        return output

    # Default: TorchScript
    print(f"  Exporting to TorchScript (imgsz={imgsz})...")
    exported = model.export(format="torchscript", imgsz=imgsz)
    print(f"  Exported to: {exported}")

    if exported and os.path.exists(exported) and exported != output:
        os.rename(exported, output)
        print(f"  Moved to: {output}")

    # Verify
    print(f"  Verifying TorchScript model...")
    m = torch.jit.load(output)
    dummy = torch.randn(1, 3, imgsz, imgsz)
    with torch.no_grad():
        out = m(dummy)
    if isinstance(out, (tuple, list)):
        out = out[0]
    print(f"  Output shape: {out.shape}")
    print(f"  Size: {os.path.getsize(output) / 1024 / 1024:.1f} MB")
    return output


def convert_onnx(source: str, output: str):
    """Convert an ONNX model to TorchScript via onnx2torch."""
    try:
        import onnx
        from onnx2torch import convert as onnx2torch_convert
    except ImportError:
        sys.exit("pip install onnx onnx2torch")

    print(f"  Loading ONNX model: {source}")
    onnx_model = onnx.load(source)

    print(f"  Converting to PyTorch...")
    pytorch_model = onnx2torch_convert(onnx_model)
    pytorch_model.eval()

    # Trace to TorchScript
    print(f"  Tracing to TorchScript...")
    # Infer input shape from ONNX model
    input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
    if any(d <= 0 for d in input_shape):
        input_shape = [1, 3, 640, 640]  # default YOLO input
    print(f"  Input shape: {input_shape}")

    dummy = torch.randn(*input_shape)
    with torch.no_grad():
        traced = torch.jit.trace(pytorch_model, dummy)

    traced.save(output)
    print(f"  Saved: {output} ({os.path.getsize(output) / 1024 / 1024:.1f} MB)")
    return output


def convert_single(source: str, output: str, imgsz: int = 640, fmt: str = "torchscript"):
    """Auto-detect source type and convert."""
    if source.endswith(".onnx") and fmt == "torchscript":
        # ONNX-to-TorchScript conversion via onnx2torch
        return convert_onnx(source, output)
    elif source.endswith(".pt") or source.endswith(".pth"):
        return convert_ultralytics(source, output, imgsz, fmt=fmt)
    elif not os.path.exists(source):
        # Assume it's a model name like "yolo11n"
        return convert_ultralytics(source, output, imgsz, fmt=fmt)
    else:
        sys.exit(f"Unknown source format: {source}")


def main():
    p = argparse.ArgumentParser(description="Convert models to TorchScript or ONNX for infergo")
    p.add_argument("--source", help="Source model (ultralytics name, .pt path, or .onnx path)")
    p.add_argument("--output", help="Output path")
    p.add_argument("--format", default="torchscript", choices=["torchscript", "onnx"],
                   help="Export format: torchscript (default) or onnx")
    p.add_argument("--batch", help="Comma-separated model names for batch conversion")
    p.add_argument("--output-dir", default="models/", help="Output directory for batch conversion")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    args = p.parse_args()

    if args.batch:
        names = [n.strip() for n in args.batch.split(",")]
        os.makedirs(args.output_dir, exist_ok=True)
        for name in names:
            ext = ".onnx" if args.format == "onnx" else ".torchscript.pt"
            out = os.path.join(args.output_dir, f"{name}{ext}")
            print(f"\n[{name}]")
            t0 = time.time()
            convert_single(name, out, args.imgsz, fmt=args.format)
            print(f"  Done in {time.time() - t0:.1f}s")
    elif args.source:
        if not args.output:
            base = os.path.splitext(os.path.basename(args.source))[0]
            ext = ".onnx" if args.format == "onnx" else ".torchscript.pt"
            args.output = f"models/{base}{ext}"
        convert_single(args.source, args.output, args.imgsz, fmt=args.format)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
