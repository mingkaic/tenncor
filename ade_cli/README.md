# ADE CLI (ADE Command Line Interface)

CLI tests the shape validation and ade equation formation

## Usage

Execute without specifying a file to enter interactive mode. Otherwise read file and print to stdout.
Specify multiple files to execute them in series (they all share the same scope).

# Grammar

Construct equations using built-in operators

The only primitive supported is an integer array formatted as [integer, ...]

Any expression can be assigned to variables. Variables are limited to 32 alphanumeric and underscore characters. Additionally, variables must start as letters or an underscore.

Each statement is executed per new line

## Built-in operators

CLI supports functions abs, sin, cos, tan, exp, log, sqrt, round, flip, pow, binomial, uniform, normal, n_elems, n_dims, argmax, rmax, rsum, permute, extend, reshape, matmul, and common operators (excluding bit manipulation and comparison operators)

## Special operators

Generate the derivative equation with respect to a variable by running `grad(expr, variable)`.

Print the shape of an expression or variable by specify it on a single line without assignment

Print the graph of an expression or variable by running `show_eq(expr)` or `show_eq(variable)`.

Exit the interactive mode or prematurely terminate file read by add `exit` line.
