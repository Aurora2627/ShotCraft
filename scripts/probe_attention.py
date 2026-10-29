"""Probe locally available SDPA kernels without loading model weights."""

import json
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend


def main():
    result = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "kernels": {},
    }
    q = torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.bfloat16)
    for name in ["FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH"]:
        try:
            with sdpa_kernel(getattr(SDPBackend, name)):
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
                torch.cuda.synchronize()
            result["kernels"][name] = "available_for_probe_shape"
        except Exception as error:
            result["kernels"][name] = type(error).__name__ + ": " + str(error)
    output = Path(__file__).resolve().parents[1] / "reports/memory-v1"
    output.mkdir(exist_ok=True)
    (output / "attention-probe.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
