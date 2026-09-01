#!/usr/bin/env python3
"""Regression test: ordinary SI-SDR is sign-invariant; positive SI-SDR is not."""

import numpy as np


def sisdr_np(y, yhat, eps=1e-8):
    y = y - y.mean()
    x = yhat - yhat.mean()
    alpha = np.sum(x * y) / (np.sum(y * y) + eps)
    target = alpha * y
    err = x - target
    return 10.0 * np.log10((np.sum(target * target) + eps) / (np.sum(err * err) + eps))


def positive_sisdr_np(y, yhat, eps=1e-8):
    y = y - y.mean()
    x = yhat - yhat.mean()
    alpha = max(np.sum(x * y) / (np.sum(y * y) + eps), eps)
    target = alpha * y
    err = x - target
    return 10.0 * np.log10((np.sum(target * target) + eps) / (np.sum(err * err) + eps))


def main():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(32768)
    pos = sisdr_np(y, y)
    neg = sisdr_np(y, -y)
    ppos = positive_sisdr_np(y, y)
    pneg = positive_sisdr_np(y, -y)
    print({"sisdr_pos": pos, "sisdr_neg": neg, "positive_sisdr_pos": ppos, "positive_sisdr_neg": pneg})
    assert abs(pos - neg) < 1e-6, "ordinary SI-SDR should be sign-invariant"
    assert ppos - pneg > 100.0, "positive-scale SI-SDR should strongly penalize polarity inversion"


if __name__ == "__main__":
    main()
