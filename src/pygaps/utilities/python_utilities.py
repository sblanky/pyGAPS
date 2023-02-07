"""Collections of various python utilities."""

import importlib
import sys
import warnings
from collections import abc


def _one_pass(iters):
    i = 0
    while i < len(iters):
        try:
            yield next(iters[i])
        except StopIteration:
            del iters[i]
        else:
            i += 1


def zip_varlen(*iterables):
    """Variable length zip() function."""
    iters = [iter(it) for it in iterables]
    while True:  # broken when an empty tuple is given by _one_pass
        val = tuple(_one_pass(iters))
        if val:
            yield val
        else:
            break


def grouped(iterable, n):
    """Divide an iterable in subgroups of max n elements."""
    return zip_varlen(*[iter(iterable)] * n)


def deep_merge(a, b, path=None, update=True):
    """Recursive updates of a dictionary."""
    if path is None:
        path = []
    for key, val in b.items():
        if key in a:
            if (isinstance(a[key], abc.Mapping) and isinstance(val, abc.Mapping)):
                deep_merge(a[key], val, path + [str(key)], update)
            elif a[key] == val:
                pass  # same leaf value
            elif update:
                a[key] = val
            else:
                raise Exception(f"Conflict at {'.'.join(path + [str(key)])}")
        else:
            a[key] = val
    return a


class SimpleWarning():
    """
    Context manager overrides warning formatter to remove unneeded info.
    """
    old_formatter = None

    def __enter__(self):
        # ignore everything except the message
        def custom_formatwarning(msg, *args, **kwargs):
            return str(msg) + '\n'

        self.old_formatter = warnings.formatwarning
        warnings.formatwarning = custom_formatwarning
        return True

    def __exit__(self, typ, value, traceback):
        warnings.formatwarning = self.old_formatter
        return True


def _load_lazy(fullname):
    """
    This lazy load was used for non-critical modules to speed import time.
    Examples: matplotlib, scipy.optimize.

    However it tends to destroy the import system. Do not use.
    """
    try:
        return sys.modules[fullname]
    except KeyError as err:
        spec = importlib.util.find_spec(fullname)
        if not spec:
            raise ModuleNotFoundError(f"Could not import {fullname}.") from err
        loader = importlib.util.LazyLoader(spec.loader)
        module = importlib.util.module_from_spec(spec)
        # Make module with proper locking and get it inserted into sys.modules.
        loader.exec_module(module)
        sys.modules[fullname] = module
        return module
