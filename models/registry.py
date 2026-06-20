AVAILABLE_MODELS = ("adasurvmamba",)


def build_model(args):
    """Return the requested survival model.

    Model source code is intentionally withheld from this public skeleton.
    """
    if args.model_type not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model_type {args.model_type!r}. Available: {AVAILABLE_MODELS}.")

    raise NotImplementedError(
        "AdaSurvMamba model implementations are not included in this public skeleton. "
        "They will be released after review and refactoring."
    )
