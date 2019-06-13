# Optimization Rules Explained in Natural Language

There are only 2 types of statements in .rules minilanguage:

1. symbol declaration
2. conversion declaration

### Comments

Only single line, double slash (//), comments are supported.

### Symbol Declaration

A symbol is a generic representation of any node in an ADE graph. Symbols must be declared before they can be used in conversions.

#### Syntax:

Symbol declaration has the following syntax:
```
symbol A // this declares A
```

### Conversion Declaration

A conversion identifies an ADE subgraph and defines a new subgraph to convert to given specied symbols and scalars.

#### Syntax:

Conversion statement has the following syntax:
```
<Subgraph to identify> => <Subgraph to convert>
```

A subgraph can be:

- a scalar (of double type). e.g.: `1`, or `2.1`
- a symbol. e.g.: `A`
- a functor. e.g.: `Function(<arguments>)`
- a group (similar to a functor, except group is prefixed with `group:`)

An argument is a subgraph with an optional edge affix `=<edge info>`

Edge info is a JSON object that contain keys `coorder` or `shaper`. Edge info requires the edge from the parent to the particular argument to match the included attributes.
