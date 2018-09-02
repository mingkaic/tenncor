# ADE (Automatic Differentiation Engine)

Given an equation built from tensors (variable defined by the shape of the data) and operations, generate a derivative of the equation.

# What are shapes? And why are they here?

Shapes are arrays of integers representing how the data is sorted. A matrix is typically represented by a row and a column. In ADE, we treat row as the limit of the data along the y-axis, and the column as the data limit along the x-axis. Together, we represent a matrix by the shape [col, row].

Since equations involve shape-changing operations, we include shapes here to validate whether an equation is valid under the shape constraints. For example, a dot operation must be performed against matrices with a common dimension.

# Where is the data?

Data calculation is not included here because implementing such operations are implementation details best deferred, because optimizing operations are hard. Instead, use an external library.

# Adding more operations

Users must implement their own iTensor that performs a gradient operation (the chain rule of said operation).

# Cons

Currently, this generator makes no guarantee on the efficiency of the generated equation.
