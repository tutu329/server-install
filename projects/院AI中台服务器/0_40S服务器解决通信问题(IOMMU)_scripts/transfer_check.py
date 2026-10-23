#!/usr/bin/env python3
"""Four-GPU copy correctness check; run in the existing PyTorch container."""
import os
import sys
import torch

torch.set_num_threads(1)
print("PyTorch:", torch.__version__, flush=True)
print("PyTorch CUDA:", torch.version.cuda, flush=True)
print("Visible devices:", os.getenv("CUDA_VISIBLE_DEVICES"), flush=True)
count = torch.cuda.device_count()
print("GPU count:", count, flush=True)
if count != 4:
    raise SystemExit("Expected 4 GPUs; check container visibility/mapping.")
for i in range(count):
    p = torch.cuda.get_device_properties(i)
    uid = getattr(p, "uuid", "unavailable")
    print(f"GPU {i}: {p.name}, UUID={uid}", flush=True)

print("\n=== CUDA peer-access capability ===", flush=True)
for i in range(count):
    row = ["-" if i == j else
           str(int(torch.cuda.can_device_access_peer(i, j)))
           for j in range(count)]
    print(f"GPU {i}: {' '.join(row)}", flush=True)

failures = []

def check(label, actual, expected):
    bad = actual.ne(expected)
    nbad = int(bad.sum().item())
    if nbad:
        indexes = bad.nonzero(as_tuple=True)[0][:5]
        examples = [(int(k), int(expected[k]), int(actual[k]))
                    for k in indexes]
        print(f"FAIL {label}: mismatches={nbad}, "
              f"(index, expected, actual)={examples}", flush=True)
        failures.append(label)
        return False
    return True

print("\n=== Single-GPU checks ===", flush=True)
reference = (torch.arange(1048576, dtype=torch.int64) * 17 + 13) % 248320
for i in range(count):
    x = reference.to(f"cuda:{i}")
    torch.cuda.synchronize(i)
    copy_ok = check(f"GPU{i} roundtrip", x.cpu(), reference)
    y = x + 7
    torch.cuda.synchronize(i)
    compute_ok = check(f"GPU{i} integer compute", y.cpu(), reference + 7)
    print(f"GPU {i}: roundtrip={'PASS' if copy_ok else 'FAIL'}, "
          f"compute={'PASS' if compute_ok else 'FAIL'}", flush=True)
    del x, y

print("\n=== Directed GPU-to-GPU copy checks ===", flush=True)
for src in range(count):
    for dst in range(count):
        if src == dst:
            continue
        pair_ok = True
        print(f"Testing GPU {src} -> GPU {dst}...", flush=True)
        for n in (32, 4096, 1048576):
            for repeat in range(3):
                expected = (torch.arange(n, dtype=torch.int64) * 17
                            + 13 + repeat * 101) % 248320
                x = expected.to(f"cuda:{src}")
                torch.cuda.synchronize(src)
                y = x.to(f"cuda:{dst}")
                torch.cuda.synchronize(dst)
                label = f"{src}->{dst} direct size={n} repeat={repeat}"
                direct_ok = check(label, y.cpu(), expected)
                host = x.cpu()
                z = host.to(f"cuda:{dst}")
                torch.cuda.synchronize(dst)
                label = f"{src}->{dst} staged size={n} repeat={repeat}"
                staged_ok = check(label, z.cpu(), expected)
                pair_ok = pair_ok and direct_ok and staged_ok
                del x, y, z, host
        print(f"GPU {src} -> GPU {dst}: "
              f"{'PASS' if pair_ok else 'FAIL'}", flush=True)

print("\n=== Summary ===", flush=True)
if failures:
    print(f"FAILED checks: {len(failures)}", flush=True)
    for item in failures:
        print(item, flush=True)
else:
    print("All tested copies and integer checks PASSED.", flush=True)
    print("Basic correctness test, not a bandwidth benchmark.", flush=True)
sys.exit(1 if failures else 0)
