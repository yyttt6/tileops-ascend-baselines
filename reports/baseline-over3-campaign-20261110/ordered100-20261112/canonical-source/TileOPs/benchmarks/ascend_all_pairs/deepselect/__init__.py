"""DeepSelect providers. Importing this package never loads a device runtime."""

from functools import partial
from typing import TYPE_CHECKING, Callable

from .adapters import factory
from .semantics import PROVIDER_OPS

if TYPE_CHECKING:
    from benchmarks.ascend_baselines import Candidate

PROVIDERS: dict[str, Callable[..., "Candidate"]] = {
    op: partial(factory, op) for op in PROVIDER_OPS
}
