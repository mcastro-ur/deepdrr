def run_with_cuda_fallback(cuda_fn, cpu_fn, *args, **kwargs):
    try:
        return cuda_fn(*args, **kwargs)
    except RuntimeError as e:
        msg = str(e).lower()
        if (
            "no kernel image is available" in msg
            or "invalid device function" in msg
            or "cuda error: invalid device function" in msg
        ):
            return cpu_fn(*args, **kwargs)
        raise
