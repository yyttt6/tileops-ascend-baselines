"""Retired adapter-owned reduction tree; original vendor kernels are unchanged."""


def extrema(case):
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase(
        "Adapter-owned repeated 2x128 reduction tree retired by user. "
        "An unchanged complete original reduction API must be bound before reuse.")
