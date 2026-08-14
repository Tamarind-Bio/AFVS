"""Vendored subset of MolPAL (MIT), pinned at a403233919b7216ab0b41a625bdf28cd8661086e.

DELIBERATELY EMPTY, and deliberately PRESENT.

Present: without this file the vendored tree is a PEP-420 namespace package, and a namespace
portion is a last-resort finder. That means an installed regular `molpal` wins over this directory
even though `select_ligands` inserts it at sys.path[0], so the isolation the vendoring exists to
provide silently evaporates in any environment where molpal is installed, and
`from molpal.acquirer import Acquirer` resolves to the installed package instead. Making this a
regular package is what actually makes the sys.path insert authoritative.

Empty: upstream's `molpal/__init__.py` does `from .explorer import Explorer`, which pulls in ray,
torch and pytorch_lightning. Importing any of that on the screening head node is the exact outcome
vendoring the Acquirer was meant to avoid, so this file must never grow re-exports. Import the
submodule you want (`from molpal.acquirer import Acquirer`) and nothing else.
"""
