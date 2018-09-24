# LLO CLI

Basically ADE CLI but with data

## Differences from ADE CLI

Primitives are multi dimension arrays instead of shapes. Arrays are formatted like python arrays. The inner most array denote a single row.

Example: [[a, b, ...], [c, d, ...], ...]

Statements are terminated with a ';' instead of newline. This is to allow multiline arrays for easier visualization.

A lone expression in a statement prints the data instead of the shape. Printing shape uses the shape builtin function.
