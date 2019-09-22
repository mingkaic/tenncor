# TEQ (Tensor EQuations)

Framework for building tensor equations.

## Components

TEQ comprises of 4 types of components:

- Coordinates
- Shapes
- Leaves/Functors
- Travelers

### Coordinates

Coordinates specify entry location on some shape-defined space, and define mapping between different shapes

### Shapes

Shapes are dimensionality boundaries and map between tensor coordinates and indices of flattened array representation

### Leaves/Functors

Leaves represemt leaf variables in an equation graph. Functors represent operations and hold coordinate mapping from each argument to coordinate

### Traveler

Travelers visits nodes in an equation graph treating functors and tensors differently
