"""Local HEI simulation and synchronized demonstration collection."""

__version__ = '0.1.0'


def __getattr__(name):
    if name == 'Environment':
        from .environment import Environment
        return Environment
    if name in ('CAMERAS', 'STATE_NAMES'):
        from . import schema
        return getattr(schema, name)
    raise AttributeError(name)
