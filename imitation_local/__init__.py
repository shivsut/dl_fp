"""imitation_local: implementations of imitation_local and reward learning algorithms."""

from importlib import metadata

try:
    __version__ = metadata.version("imitation_local")
except metadata.PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass
