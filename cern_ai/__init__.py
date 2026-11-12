"""Particle Transformer for CMS NanoAOD.

A compact, from-scratch implementation of the ParT architecture
(arXiv:2206.11833) that streams CMS open data over plain HTTP and trains on
a single consumer GPU.

Quick start::

    pip install -e .
    python -m cern_ai.generate_urls
    python -m cern_ai.train --preset base --n-steps 5000
    python -m cern_ai.evaluate --checkpoint checkpoints/final.pt
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
