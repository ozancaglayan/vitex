
"""Backends for rendering."""


def get_renderer(backend: str = 'freetype'):
    if backend == 'freetype':
        from .freetype import Renderer
        return Renderer
    elif backend == 'pillow':
        from .pillow import Renderer
        return Renderer
    else:
        raise RuntimeError(f'Backend {backend} unknown.')
