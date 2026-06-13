from __future__ import annotations

import atexit
import itertools
import tempfile

import dask.array as da

# A single process-wide temporary directory holds every cached array. It's created lazily on first
# use and removed when the interpreter exits.
_cache_dir: tempfile.TemporaryDirectory | None = None
_counter = itertools.count()


def _cache_path() -> str:
    global _cache_dir
    if _cache_dir is None:
        _cache_dir = tempfile.TemporaryDirectory(prefix="magnify-cache-")
        atexit.register(_cache_dir.cleanup)
    return _cache_dir.name


def cache(array):
    """Save a dask array to an on-disk zarr store and return a lazy array backed by it.

    This behaves like :meth:`dask.array.Array.persist` but spills the result to disk instead of
    holding it in memory, which keeps the task graph small without exhausting RAM on large
    datasets. Non-dask arrays are returned unchanged.
    """
    if not isinstance(array, da.Array):
        return array
    # Each array goes to a unique component so caching never overwrites a store that an existing
    # array is still reading from.
    component = f"array-{next(_counter)}"
    return da.to_zarr(array, url=_cache_path(), component=component, return_stored=True)
