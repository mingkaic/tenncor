///
/// api.hpp
/// llo
///
/// Purpose:
/// Define API to easily build equations
///

#include "llo/node.hpp"

#ifndef LLO_API_HPP
#define LLO_API_HPP

namespace llo
{

/// Returns DataNode with SYMBOLIC_ONE
DataNode one (void);

/// Element-wise absolute operation
DataNode abs (DataNode arg);

/// Element-wise negative operation
DataNode neg (DataNode arg);

/// Element-wise bitwise not operation
DataNode bit_not (DataNode arg);

/// Element-wise sine operation
DataNode sin (DataNode arg);

/// Element-wise cosine operation
DataNode cos (DataNode arg);

/// Element-wise tangent operation
DataNode tan (DataNode arg);

/// Element-wise exponent operation
DataNode exp (DataNode arg);

/// Element-wise natural log operation
DataNode log (DataNode arg);

/// Element-wise square root operation
DataNode sqrt (DataNode arg);

/// Element-wise round operation
DataNode round (DataNode arg);

/// Flip values of arg along specified dimension
DataNode flip (DataNode arg, uint8_t dim);

/// Element-wise operation: base ^ exponent
DataNode pow (DataNode a, DataNode b);

/// Element-wise operation: a + b
DataNode add (DataNode a, DataNode b);

/// Get Element-wise sum of args
DataNode sum (std::vector<DataNode> args);

/// Element-wise operation: a - b
DataNode sub (DataNode a, DataNode b);

/// Element-wise operation: a///b
DataNode mul (DataNode a, DataNode b);

/// Get Element-wise product of args
DataNode prod (std::vector<DataNode> args);

/// Element-wise operation: a / b
DataNode div (DataNode a, DataNode b);

/// Element-wise operation: a == b
DataNode eq (DataNode a, DataNode b);

/// Element-wise operation: a != b
DataNode neq (DataNode a, DataNode b);

/// Element-wise operation: a < b
DataNode lt (DataNode a, DataNode b);

/// Element-wise operation: a > b
DataNode gt (DataNode a, DataNode b);

/// Get Element-wise minimum of args
DataNode min (std::vector<DataNode> args);

/// Get Element-wise maximum of args
DataNode max (std::vector<DataNode> args);

/// Element-wise clip x between lo and hi
/// Values in x larger than hi take value hi, vice versa for lo
DataNode clip (DataNode x, DataNode lo, DataNode hi);

/// Generate random numbers according to std::binomial_distribution(a, b)
DataNode rand_binom (DataNode ntrials, DataNode prob);

/// Generate random numbers according to std::uniform_distributon(a, b)
DataNode rand_uniform (DataNode lower, DataNode upper);

/// Generate random numbers according to std::normal_distribution(a, b)
DataNode rand_normal (DataNode mean, DataNode stdev);

/// Get n_elem of input shape as value
DataNode n_elems (DataNode arg);

/// Get value at specified dimension of input shape
DataNode n_dims (DataNode arg, uint8_t dim);

/// Get first flat index of the max value
DataNode argmax (DataNode arg);

/// Get the max value
DataNode reduce_max (DataNode arg);

/// Get the sum of all values
DataNode reduce_sum (DataNode arg);

/// Apply reduce_sum to elements of coordinate range [0:groupidx]
DataNode reduce_sum (DataNode arg, uint8_t groupidx);

/// Matrix multiply 2 or 1 dimension matrices,
/// Tensors with ranks higher than 2 throws runtime error
DataNode matmul (DataNode a, DataNode b);

// // NOT IMPLEMENTED
// DataNode convolute (DataNode canvas, DataNode window);

/// Permute shape according to input indices. output shape take
/// on input dimensions ordered by indices, and concatenated by unreferenced
/// input dimensions ordered by input's original order
DataNode permute (DataNode arg, std::vector<uint8_t> order);

/// Concatenate input shape vector to input tensor's shape.
/// expect value to expand into the new shape by duplicating
DataNode extend (DataNode arg, std::vector<uint8_t> ext);

}

#endif // LLO_API_HPP
