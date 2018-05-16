# clay

clay library is responsible for managing the shape and type of tensor data

There are 3 concrete classes and 1 enum:

- Tensor: principle component containing Shape and DTYPE
- Shape: vector of unsigned integers with each element representing a dimensional value
- State: transactional component composing weak reference to data, and shape and type held by Tensor
- DTYPE: enumerating the supported data types

Tensor relies on a source interface to dynamically update raw data, shape, and type

Additionally, clay supplies an abstract builder, and type independent memory allocation and type mapping utility functions.
