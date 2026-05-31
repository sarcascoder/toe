"""PORE: Partial-Order Reading-order Evaluation.

A research artifact for measuring document reading-order quality in a way that
(1) is robust to legitimate reading-order ambiguity (multi-column, sidebars,
footnotes), and (2) orthogonally decomposes parsing error into a
transcription component and an ordering component.

Modules:
    partial_order - partial-order (DAG) representation + violation rate
    metric        - PORE decomposition over matched blocks
    synth         - synthetic benchmark generator with ground-truth DAGs
    run_study     - evaluate models / mocks and produce figures
"""

__version__ = "0.1.0"
