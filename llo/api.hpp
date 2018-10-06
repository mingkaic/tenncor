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

/// Element-wise absolute operation
ade::Tensorptr abs (ade::Tensorptr arg);

/// Element-wise negative operation
ade::Tensorptr neg (ade::Tensorptr arg);

/// Element-wise bitwise not operation
ade::Tensorptr bit_not (ade::Tensorptr arg);

/// Element-wise sine operation
ade::Tensorptr sin (ade::Tensorptr arg);

/// Element-wise cosine operation
ade::Tensorptr cos (ade::Tensorptr arg);

/// Element-wise tangent operation
ade::Tensorptr tan (ade::Tensorptr arg);

/// Element-wise exponent operation
ade::Tensorptr exp (ade::Tensorptr arg);

/// Element-wise natural log operation
ade::Tensorptr log (ade::Tensorptr arg);

/// Element-wise square root operation
ade::Tensorptr sqrt (ade::Tensorptr arg);

/// Element-wise round operation
ade::Tensorptr round (ade::Tensorptr arg);

/// Flip values of arg along specified dimension
ade::Tensorptr flip (ade::Tensorptr arg, uint8_t dim);

/// Element-wise operation: base ^ exponent
ade::Tensorptr pow (ade::Tensorptr a, ade::Tensorptr b);

/// Element-wise operation: a + b
ade::Tensorptr add (ade::Tensorptr a, ade::Tensorptr b);

/// Get Element-wise sum of args
ade::Tensorptr sum (std::vector<ade::Tensorptr> args);

/// Element-wise operation: a - b
ade::Tensorptr sub (ade::Tensorptr a, ade::Tensorptr b);

/// Element-wise operation: a///b
ade::Tensorptr mul (ade::Tensorptr a, ade::Tensorptr b);

/// Get Element-wise product of args
ade::Tensorptr prod (std::vector<ade::Tensorptr> args);

/// Element-wise operation: a / b
ade::Tensorptr div (ade::Tensorptr a, ade::Tensorptr b);

/// Element-wise operation: a == b
ade::Tensorptr eq (ade::Tensorptr a, ade::Tensorptr b);

/// Element-wise operation: a != b
ade::Tensorptr neq (ade::Tensorptr a, ade::Tensorptr b);

/// Element-wise operation: a < b
ade::Tensorptr lt (ade::Tensorptr a, ade::Tensorptr b);

/// Element-wise operation: a > b
ade::Tensorptr gt (ade::Tensorptr a, ade::Tensorptr b);

/// Get Element-wise minimum of args
ade::Tensorptr min (std::vector<ade::Tensorptr> args);

/// Get Element-wise maximum of args
ade::Tensorptr max (std::vector<ade::Tensorptr> args);

/// Element-wise clip x between lo and hi
/// Values in x larger than hi take value hi, vice versa for lo
ade::Tensorptr clip (ade::Tensorptr x, ade::Tensorptr lo, ade::Tensorptr hi);

/// Generate random numbers according to std::binomial_distribution(a, b)
ade::Tensorptr rand_binom (ade::Tensorptr ntrials, ade::Tensorptr prob);

/// Generate random numbers according to std::uniform_distributon(a, b)
ade::Tensorptr rand_uniform (ade::Tensorptr lower, ade::Tensorptr upper);

/// Generate random numbers according to std::normal_distribution(a, b)
ade::Tensorptr rand_normal (ade::Tensorptr mean, ade::Tensorptr stdev);

/// Get n_elem of input shape as value
ade::Tensorptr n_elems (ade::Tensorptr arg);

/// Get value at specified dimension of input shape
ade::Tensorptr n_dims (ade::Tensorptr arg, uint8_t dim);

/// Get first flat index of the max value
ade::Tensorptr argmax (ade::Tensorptr arg);

/// Get the max value
ade::Tensorptr reduce_max (ade::Tensorptr arg);

/// Get the sum of all values
ade::Tensorptr reduce_sum (ade::Tensorptr arg);

/// Matrix multiply 2 or 1 dimension matrices,
/// Tensors with ranks higher than 2 throws runtime error
ade::Tensorptr matmul (ade::Tensorptr a, ade::Tensorptr b);

/// High dimension matrix multiplication, using 2 group indices,
/// for each tensor, form groups [:idx) and [index:rank) and treat dimensions
/// falling in those ranges as a single dimension (where the shape values must
/// match) then apply matmul given the grouped shape
/// For example, given shapea={3, 4, 5}, ai=2, shapeb={7, 8, 3, 4}, bi=2,
/// output tensor has shape {7, 8, 5}, since {3, 4} in a and b matches
ade::Tensorptr matmul (ade::Tensorptr a, ade::Tensorptr b,
	uint8_t agroup_idx, uint8_t bgroup_idx);

// // NOT IMPLEMENTED
// ade::Tensorptr convolute (ade::Tensorptr canvas, ade::Tensorptr window);

/// Permute shape according to input indices. output shape take
/// on input dimensions ordered by indices, and concatenated by unreferenced
/// input dimensions ordered by input's original order
ade::Tensorptr permute (ade::Tensorptr arg, std::vector<uint8_t> order);

/// Concatenate input shape vector to input tensor's shape.
/// expect value to expand into the new shape by duplicating
ade::Tensorptr extend (ade::Tensorptr arg, std::vector<uint8_t> ext);

/// Reshape input tensor's shape to new shape assuming the new
/// shape has the same n_elems as old shape
ade::Tensorptr reshape (ade::Tensorptr arg, std::vector<uint8_t> slist);

// // NOT IMPLEMENTED
// ade::Tensorptr group (ade::OPCODE op, std::vector<std::pair<ade::Tensorptr,std::vector<uint8_t>> args);

}

#endif // LLO_API_HPP
