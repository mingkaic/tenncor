# Optimization Rules Explained in Natural Language

### Comments

Single line comment are double slashes (//).

## Statements

There are only 3 types of statements in .rules minilanguage:

- symbol declaration
- group declaration
- conversion declaration

### Symbol Declaration

A symbol is a generic representation of any node in an ADE graph.
In conversions, symbols can be used to represent "leaves" of subgraphs.
Symbols must be declared before they can be used in conversions.

#### Syntax:

Symbol declaration has the following syntax:
```
symbol A // this declares A
```

### Group Declaration

A group is a collective of functors in an ADE graph.
In conversions, group label associates the source subgraph to the target subgraph in the association when there are multiple groups. Additionally, group tag denotes the tag used to identify functors of the group in the ADE graph.
Groups must be declared before they can be used in conversions.

#### Syntax:

Group declaration has the following syntax:
```
groupdef X group1 // this declares X label using tag group1
groupdef Y group1

group X(group Y(A,C),B) => group Y(group X(A,B),C) // Y group1 is now above X group1
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
- a group. e.g.: `group G(<arguments>)`

An argument is a subgraph with an optional edge affix `=<edge info>`

Edge info is a JSON object that contain keys `coorder` or `shaper`. Edge info requires the edge from the parent to the particular argument to match the included attributes.

A group is allowed one variadic argument at the end of arguments list (e.g.: `group G(<arguments>,...B)`). This argument cannot be a subgraph and it is not allowed edge affix.
