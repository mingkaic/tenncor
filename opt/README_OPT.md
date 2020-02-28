# OPT (OPTimization)

This module implements subgraph optimization using query module

# Matching Methods

Optimization rules are specified by objects with `srcs` and `dest` fields.

`srcs` is an array of the operation subgraph patterns to match

`dest` is an operation subgraph into which `srcs` are converted

## Subgraphs, Variables, Scalars

Rule `srcs` patterns exactly follows the query module's schema.

Toplevel object is a node which can take on `op`, `var`, `cst`, or `symb`.

- `op` matches against functors

- `var` matches against leaves

- `cst` matches against scalar constant leaves

- `symb` matches against any node

## Any Node Selection

Subgraphs can be ambiguously selected by the `symb` Node field. These fields can be referenced in rule's `dest` field.

For example:

given a source of `{"op": "SIN", "args": [{"symb": "X"}]}`
and a dest of `{"symb": "X"}`

`tenncor::add(tenncor::sin(a),tenncor::cos(b))` will be transformed to
`tenncor::add(a,tenncor::cos(b))`
