"""Bake the retrieval models into an image at the revisions the index was verified with.

The Docker build runs this once so the running service never downloads a model:
the Space starts with ``HF_HUB_OFFLINE=1`` and loads both models from this cache.
The revisions are the ones cached on the machine that built and verified the
committed index. A different revision of the embedding model would embed queries
into a different space from the stored vectors, and nothing would fail loudly.

``--verify`` loads both models the way the service does, offline, so a missing file
fails the image build instead of the first visitor's question.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# repo -> (revision, the one weights file that revision ships). Each repository is
# fetched with exactly one weights file: bge-large-zh-v1.5 has only a PyTorch pickle
# at this revision (a desktop's safetensors copy comes from the Hub's automatic
# conversion branch, which offline loading never consults), while the reranker has
# both formats and would otherwise double its size.
MODELS = {
    "BAAI/bge-large-zh-v1.5": (
        "79e7739b6ab944e86d6171e44d24c997fc1e0116",
        "pytorch_model.bin",
    ),
    "BAAI/bge-reranker-base": (
        "2cfc18c9415c912f9d8155881c133215df768a70",
        "model.safetensors",
    ),
}
WEIGHT_FILES = ["*.bin", "*.safetensors", "*.pt", "*.h5", "*.msgpack", "*.ot", "*.onnx", "onnx/*"]


def fetch_models() -> None:
    from huggingface_hub import constants, hf_hub_download, snapshot_download

    for repo_id, (revision, weights) in MODELS.items():
        snapshot_download(repo_id, revision=revision, ignore_patterns=WEIGHT_FILES)
        hf_hub_download(repo_id, weights, revision=revision)
        # Loading by name resolves "main" through refs/main, and a download by commit
        # hash does not write that ref, so offline loading would otherwise fail.
        ref = Path(constants.HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
        ref = ref / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(revision, encoding="utf-8")
        print(f"{repo_id}@{revision[:12]} 已缓存（{weights}）")


def verify_offline() -> None:
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SystemExit("--verify 必须在 HF_HUB_OFFLINE=1 下运行，否则缺失的文件会被悄悄下载")
    from sentence_transformers import CrossEncoder, SentenceTransformer

    embedding, reranker = MODELS
    # Same constructors, arguments and device as providers/local.py on a CPU host.
    vector = SentenceTransformer(embedding, device="cpu").encode(["进程"])[0]
    score = CrossEncoder(reranker, device="cpu").predict([("进程", "进程是程序的一次执行")])[0]
    print(f"离线加载通过：embedding 维度 {len(vector)}，reranker 分数 {float(score):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true", help="离线加载已缓存的模型并试算一次")
    if parser.parse_args().verify:
        verify_offline()
    else:
        fetch_models()
