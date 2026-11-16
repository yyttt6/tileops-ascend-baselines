"""The complete native ops-fft C2C provider, including host transfer costs."""


def fft(case):
    from benchmarks.ascend_baselines import opsfft_fft

    return opsfft_fft(case)


PROVIDERS = {"FFTC2CFwdOp": fft}
