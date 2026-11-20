"""BitRoss 2: text-to-16x16 Minecraft texture generation with masked discrete diffusion."""

__version__ = "2.0.0"


def __getattr__(name):
    if name == "BitRoss":
        from .pipeline import BitRoss
        return BitRoss
    raise AttributeError(name)
