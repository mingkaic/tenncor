# Optimization Rules Explained in Natural Language

### Comments

Single line comment are double slashes (//).

## Statements

There are 4 types of statements in .rules minilanguage:

- symbol declaration
- group declaration
- property mapping
- conversion

## Symbol Declaration

A symbol is a generic representation of any node in an TEQ graph.
In conversions, symbols can be used to represent "leaves" of subgraphs.
Symbols must be declared before they can be used in conversions.

#### Syntax:

Symbol declaration has the following syntax:
```
symbol A // this declares A
```

## Property Mapping

## Conversion

A conversion identifies an TEQ subgraph and defines a new subgraph to convert to given specied symbols and scalars.

#### Syntax:

Conversion statement has the following syntax:
```
<Subgraph to identify> => <Subgraph to convert>
```

A subgraph can be:

- a scalar (of double type). e.g.: `1`, or `2.1`
- a symbol. e.g.: `A`
- a functor. e.g.: `Function(<arguments>)`
- a group. e.g.: `group:G(<arguments>)`

An argument is a subgraph with an optional edge affix `=<edge info>`

Edge info is a JSON object that contain keys `coorder` or `shaper`. Edge info requires the edge from the parent to the particular argument to match the included attributes.

A group is allowed one variadic argument at the end of arguments list (e.g.: `group:G(<arguments>,...B)`). This argument cannot be a subgraph and it is not allowed edge affix.
