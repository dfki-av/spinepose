import concurrent.futures
from typing import Any, Callable, Iterable, List, Optional


def concurrent_forloop(
    func: Callable[..., Any],
    iterable: Iterable,
    *iterables: Iterable,
    max_workers: Optional[int] = None,
) -> List[Any]:
    """Runs a function concurrently over one or more iterables.

    Args:
        func: Function to run for each item.
        iterable: Primary iterable of inputs.
        *iterables: Additional iterables zipped with the primary iterable.
        max_workers: Maximum number of worker threads.

    Returns:
        List[Any]: Results in input order.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, iterable, *iterables))
