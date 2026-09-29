#!/usr/bin/env python3
"""Download Qwen3.8-27B Q4_0 + its MTP head from Hugging Face with resume and SHA-256 check.

    python3 scripts/download-models.py [models_dir]

Uses only the standard library and curl (resume, retries). Set HF_ENDPOINT for a mirror,
HF_TOKEN if you need authentication.
"""
import hashlib
import json
import os
import subprocess
import sys
import urllib.request

REPO = "unsloth/Qwen3.8-27B-GGUF"
FILES = {  # file in repo -> local name
    "Qwen3.8-27B-Q4_0.gguf": "Qwen3.8-27B-Q4_0.gguf",
    "MTP/mtp-Qwen3.8-27B-Q4_0.gguf": "mtp-Qwen3.8-27B-Q4_0.gguf",
}
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
TOKEN = os.environ.get("HF_TOKEN", "")
dest = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "models")
os.makedirs(dest, exist_ok=True)


def api_info():
    req = urllib.request.Request(f"{ENDPOINT}/api/models/{REPO}?blobs=true")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    return {s["rfilename"]: s for s in data.get("siblings", [])}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


info = api_info()
for remote, local in FILES.items():
    meta = info.get(remote)
    if meta is None:
        sys.exit(f"{remote} not found in {REPO} - the repo layout changed, please open an issue")
    size = meta.get("size") or (meta.get("lfs") or {}).get("size")
    want = (meta.get("lfs") or {}).get("sha256")
    path = os.path.join(dest, local)
    done = path + ".sha256-ok"
    if os.path.exists(done) and os.path.getsize(path) == size:
        print(f"{local}: already downloaded and verified")
        continue
    print(f"{local}: {size / 1e9:.2f} GB from {REPO}")
    cmd = ["curl", "-L", "--fail", "-C", "-", "--retry", "10", "--retry-delay", "5",
           "-o", path, f"{ENDPOINT}/{REPO}/resolve/main/{remote}"]
    if TOKEN:
        cmd[1:1] = ["-H", f"Authorization: Bearer {TOKEN}"]
    rc = subprocess.call(cmd)
    if rc not in (0, 33):  # 33: range not satisfiable = already complete
        sys.exit(f"{local}: curl exited with {rc}; run again to resume")
    if size and os.path.getsize(path) != size:
        sys.exit(f"{local}: size {os.path.getsize(path)} != {size}; run again to resume")
    if want:
        print(f"{local}: checking SHA-256 ...")
        got = sha256(path)
        if got != want:
            os.rename(path, path + ".bad")
            sys.exit(f"{local}: SHA-256 mismatch (moved to .bad), run again")
    open(done, "w").write(want or "")
    print(f"{local}: OK")
print(f"Models are in {dest}")
