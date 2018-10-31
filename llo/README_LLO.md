# LLO (Low-Level Operators)

Using ADE, construct an equation graph, wrap data-relevant arguments missing in ADE nodes (e.g.: flip dimension), then map the nodes to a low level implementation of the operator mapped by the node.

LLO provides an evaluation function which directs the appropriate arguments to the mapped operation implementation.

# Future Work

In order for decouple the ADE extension from the operator implementations, we need to experiment and settle on a stable API.
This module is ultimately a wrapper library for ADE, and operator libraries for the benefit of top-level executables.
