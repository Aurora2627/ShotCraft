"""Bound memory for the adapter's single-window, no-ID-cache inference path."""

from contextlib import contextmanager


@contextmanager
def disable_single_window_feature_cache(transformer):
    """Do not collect per-layer states; preserve the upstream tuple contract.

    Only valid for one window, prev_clip_weight=0 and no ID resampling.
    Does not change model weights or attention math; hooks are temporary.
    """

    def before(module, args, kwargs):
        if kwargs.get("return_dict", True):
            raise ValueError("Single-window adapter requires tuple outputs")
        if (
            kwargs.get("id_pool_resample_learnable", False)
            or (kwargs.get("attention_kwargs") or {}).get("prev_hidden_states")
            is not None
        ):
            raise ValueError(
                "Cannot suppress features needed by cross-window/ID inference"
            )
        kwargs = dict(kwargs, return_hidden_states=False, return_resample_mask=False)
        return args, kwargs

    def after(module, args, kwargs, output):
        if not isinstance(output, tuple) or len(output) != 1:
            raise RuntimeError("Unexpected pinned transformer return contract")
        return output[0], [], None

    pre = transformer.register_forward_pre_hook(before, with_kwargs=True)
    post = transformer.register_forward_hook(after, with_kwargs=True)
    try:
        yield
    finally:
        post.remove()
        pre.remove()
