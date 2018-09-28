/*!
 *
 *  api.hpp
 *  llo
 *
 *  Purpose:
 *  define api for easily tensors operations
 *
 */

#include "llo/node.hpp"

#ifndef LLO_API_HPP
#define LLO_API_HPP

namespace llo
{

/*! Valuewise absolute operation */
ade::Tensorptr abs (ade::Tensorptr arg);

/*! Value-wise negative operation */
ade::Tensorptr neg (ade::Tensorptr arg);

/*! Value-wise bitwise not operation */
ade::Tensorptr bit_not (ade::Tensorptr arg);

/*! Value-wise sine operation*/
ade::Tensorptr sin (ade::Tensorptr arg);

/*! Value-wise cosine operation*/
ade::Tensorptr cos (ade::Tensorptr arg);

/*! Value-wise tangent operation*/
ade::Tensorptr tan (ade::Tensorptr arg);

/*! Value-wise exponent operation*/
ade::Tensorptr exp (ade::Tensorptr arg);

/*! Value-wise natural log operation*/
ade::Tensorptr log (ade::Tensorptr arg);

/*! Value-wise square root operation*/
ade::Tensorptr sqrt (ade::Tensorptr arg);

/*! Value-wise round operation*/
ade::Tensorptr round (ade::Tensorptr arg);

/*! Flip values along a dimension */
ade::Tensorptr flip (ade::Tensorptr arg, uint8_t dim);

/*! Value-wise operation: base ^ exponent*/
ade::Tensorptr pow (ade::Tensorptr a, ade::Tensorptr b);

/*! Value-wise operation: a + b*/
ade::Tensorptr add (ade::Tensorptr a, ade::Tensorptr b);

/*! Value-wise operation: a - b*/
ade::Tensorptr sub (ade::Tensorptr a, ade::Tensorptr b);

/*! Value-wise operation: a * b*/
ade::Tensorptr mul (ade::Tensorptr a, ade::Tensorptr b);

/*! Value-wise operation: a / b*/
ade::Tensorptr div (ade::Tensorptr a, ade::Tensorptr b);

/*! Value-wise operation: a == b*/
ade::Tensorptr eq (ade::Tensorptr a, ade::Tensorptr b);

/*! Value-wise operation: a != b*/
ade::Tensorptr neq (ade::Tensorptr a, ade::Tensorptr b);

/*! Value-wise operation: a < b*/
ade::Tensorptr lt (ade::Tensorptr a, ade::Tensorptr b);

/*! Value-wise operation: a > b*/
ade::Tensorptr gt (ade::Tensorptr a, ade::Tensorptr b);

/*! Generate random numbers according to std::binomial_distribution(a, b) */
ade::Tensorptr binom (ade::Tensorptr ntrials, ade::Tensorptr prob);

/*! Generate random numbers according to std::uniform_distributon(a, b) */
ade::Tensorptr uniform (ade::Tensorptr lower, ade::Tensorptr upper);

/*! Generate random numbers according to std::normal_distribution(a, b) */
ade::Tensorptr normal (ade::Tensorptr mean, ade::Tensorptr stdev);

/*! Get n_elem of input shape as value */
ade::Tensorptr n_elems (ade::Tensorptr arg);

/*! Get value at specified dimension of input shape */
ade::Tensorptr n_dims (ade::Tensorptr arg, uint8_t dim);

/*! Get first flat index of the max value */
ade::Tensorptr argmax (ade::Tensorptr arg);

/*! Get the max value */
ade::Tensorptr rmax (ade::Tensorptr arg);

/*! Get the sum of all values */
ade::Tensorptr rsum (ade::Tensorptr arg);

/*! Matrix multiplication of 2 or 1 dimension matrices,
 *  higher dimensions throws runtime error */
ade::Tensorptr matmul (ade::Tensorptr a, ade::Tensorptr b);

/*! High dimension matrix multiplication, translating input into 2-D virtual
 *  matrices, take n_elems of subshape in range [:group_idx] as the first
 *  dimension and n_elems of subshape in range [group_idx:] as second dimension
 */
ade::Tensorptr matmul (ade::Tensorptr a, ade::Tensorptr b,
	uint8_t agroup_idx, uint8_t bgroup_idx);

// NOT IMPLEMENTED
ade::Tensorptr convolute (ade::Tensorptr canvas, ade::Tensorptr window);

/*! Permute shape according to input indices. output shape take
 *  on input dimensions ordered by indices, and concatenated by unreferenced
 *  input dimensions ordered by input's original order */
ade::Tensorptr permute (ade::Tensorptr arg, std::vector<uint8_t> order);

/*! Concatenate input shape vector to input tensor's shape.
 *  expect value to expand into the new shape by duplicating */
ade::Tensorptr extend (ade::Tensorptr arg, std::vector<uint8_t> ext);

/*! Reshape input tensor's shape to new shape assuming the new
 *  shape has the same n_elems as old shape */
ade::Tensorptr reshape (ade::Tensorptr arg, std::vector<uint8_t> slist);

}

#endif /* LLO_API_HPP */
