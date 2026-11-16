"""Lazy tilelang-ascend providers. Importing this entry performs no device imports."""
from functools import partial
from typing import TYPE_CHECKING, Callable

from .plans import IMPLEMENTED

if TYPE_CHECKING:
    from benchmarks.api import Case
    from benchmarks.ascend_baselines import Candidate


def _make(op,case):
    from .native import prepare_deployment
    prepare_deployment()
    from .adapters import factory
    return factory(op,case)


PROVIDERS: dict[str, Callable[['Case'], 'Candidate']] = {
    op: partial(_make,op) for op in sorted(IMPLEMENTED)
}

from .integration700_pto23 import install as _install_pto23_700
_install_pto23_700(PROVIDERS)
