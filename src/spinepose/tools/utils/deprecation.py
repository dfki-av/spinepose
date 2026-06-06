from __future__ import annotations

import warnings
from functools import wraps
from typing import Any


def deprecation_warning(message: str):
    """Issue a deprecation warning.

    Args:
        message: Message value.
    """
    warnings.warn(message, DeprecationWarning, stacklevel=2)


def deprecated_arg(
    old_arg: str,
    new_arg: str | None = None,
    default: Any = None,
):
    """Decorator to mark a function argument as deprecated.

    Args:
        old_arg: Old argument name.
        new_arg: Optional new argument name.
        default: Default value for the new argument, if specified.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if old_arg in kwargs:
                # Emit a deprecation warning
                message = f"The '{old_arg}' argument is deprecated."
                if new_arg:
                    message += f" Please use '{new_arg}' instead."
                deprecation_warning(message)

                old_value = kwargs.pop(old_arg)

                # Map to new argument if needed, but let explicit new-style args win.
                if new_arg and new_arg not in kwargs:
                    if old_arg == "device" and new_arg == "hardware_acceleration":
                        kwargs[new_arg] = old_value != "cpu"
                    elif default is not None:
                        kwargs[new_arg] = default
                    else:
                        kwargs[new_arg] = old_value
            return func(*args, **kwargs)

        return wrapper

    return decorator
