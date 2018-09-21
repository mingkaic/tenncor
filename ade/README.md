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

# Test Plan

ADE comprises of 5 components:

- Shape
- Tensor
- Fwder
- Functor
- Grader

With the exception of Grader, the remaining 4 components can have their test data auto-generated due to their expected output following an expected rule.

### Shapes

Shapes behave like an array and and hold information regarding their initial integer array

### Tensors

Tensors hold shape info, and their gradient is 1 or 0 depending on the node it's deriving with respect to

### Fwders

Forwarders aggregate and transform shape information and fall in 3 classes:

- elementary (or identity): these reject non-scalar shapes that are not compatible with other non-scalar shapes
- reshape transformer: takes a single shape and an integer array argument. This integer array holds some dimension information on how to transform the shape.
- multi-argument aggregators: such as matrix multiplication or convolution

Since all forwarders conform to a specific rule, their expected behavior can inferred from auto-generated input (mostly shapes).

### Functors

Functors wrap the components and its gradient should conform to the grader associated with its initial opcode. Tests in functor should be minimal, but still cover trivial cases getting arguments, to_string and basic gradient outputs, due to potential typos later on.

### Graders

Graders specify the chain rule mapped by a specific operation. Its behavior is adhoc, so it does not benefit from auto-generation.
Easiest test is to stringify the generated equation tree and match against test data
