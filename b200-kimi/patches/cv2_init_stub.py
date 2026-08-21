"""
LOCAL PATCH (b200-kimi). Replaces vllm:kimi-k3's cv2/__init__.py in its entirety.

cv2's real bootstrap() (the original of this file) native-loads its .so via dlopen, and
that load segfaults non-deterministically under this container's environment at high
request concurrency -- confirmed at THREE independent call sites across two Kimi-K3
2-node runs:
  1. vllm.multimodal.video               (patched separately, patches/video.py)
  2. mistral_common.imports.is_opencv_installed(), reached via
     transformers.tokenization_mistral_common -- transformers' own tokenizer
     auto-detection, triggered on every `vllm bench serve` startup regardless of the
     actual tokenizer in use.
  3. (likely others -- cv2 is a common optional dependency across the stack; patching
     every call site individually does not scale)

A segfault during `import cv2` is not a catchable Python exception, so the
`except ImportError` guards already present at every one of those call sites (vLLM's
own PlaceholderModule fallback, and mistral_common's own is_opencv_installed() check)
were never able to protect anything -- the process died before Python's exception
machinery ever got control back.

This benchmark sends no images or video (`--dataset-name random`, pure text), so cv2 is
never functionally needed. Replacing __init__.py with a clean, immediate ImportError
turns "crashes the whole process" into "the try/except that was already written
everywhere works as originally intended" -- for every current and future call site, not
just the ones found by reproducing crashes one at a time.
"""
raise ImportError(
    "cv2 disabled by local patch (b200-kimi): native bootstrap segfaults under this "
    "container's threading/mmap environment at high concurrency. Not needed for "
    "text-only benchmarking -- see patches/cv2_init_stub.py for the full story."
)
