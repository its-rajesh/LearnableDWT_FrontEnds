#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dwt_ir", type=Path, default=Path("/home/rrame12/Desktop/Research/DWT_IR"))
    ap.add_argument("--best_model", type=Path, required=True)
    ap.add_argument("--x", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--filter_length", type=int, default=11)
    ap.add_argument("--pr_shifts", type=int, default=12)
    ap.add_argument("--pr_lambda", type=float, default=0.1)
    ap.add_argument("--pr_dc_lambda", type=float, default=1.0)
    ap.add_argument("--pr_nyq_lambda", type=float, default=1.0)
    ap.add_argument("--unet_depth", type=int, default=4)
    ap.add_argument("--base_filters", type=int, default=64)
    args = ap.parse_args()

    sys.path.insert(0, str(args.dwt_ir))
    from test_iprdwt_rerecorded_updated import (  # noqa: PLC0415
        PRDWT1D,
        PRIDWT1D,
        MatchTimeLen,
        SplitChannels,
        UpsampleTo,
        build_base_model,
        load_base_model_from_any,
    )

    X = np.load(args.x).astype(np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected X shape (N,C,T), got {X.shape}")
    _, C, T = X.shape

    custom_objs = {
        "MatchTimeLen": MatchTimeLen,
        "SplitChannels": SplitChannels,
        "UpsampleTo": UpsampleTo,
        "PRDWT1D": PRDWT1D,
        "PRIDWT1D": PRIDWT1D,
    }

    def base_builder():
        return build_base_model(
            T=T,
            C=C,
            levels=args.levels,
            filt_len=args.filter_length,
            pr_shifts=args.pr_shifts,
            pr_lambda=args.pr_lambda,
            pr_dc_lambda=args.pr_dc_lambda,
            pr_nyq_lambda=args.pr_nyq_lambda,
            unet_depth=args.unet_depth,
            base_filters=args.base_filters,
        )

    model = load_base_model_from_any(
        best_model_path=str(args.best_model),
        custom_objs=custom_objs,
        base_builder=base_builder,
        C=C,
        T=T,
        levels=args.levels,
    )

    Ypred = np.zeros_like(X, dtype=np.float32)
    for start in range(0, len(X), args.batch):
        stop = min(len(X), start + args.batch)
        Ypred[start:stop] = model.predict(X[start:stop], verbose=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, Ypred.astype(np.float32))
    print("Saved:", args.out)
    print("Shape:", Ypred.shape)


if __name__ == "__main__":
    main()
