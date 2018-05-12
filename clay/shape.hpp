/*!
 *
 *  shape.hpp
 *  clay
 *
 *  Purpose:
 *  shape stores aligned dimension info
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <vector>

#pragma once
#ifndef CLAY_SHAPE_HPP
#define CLAY_SHAPE_HPP

namespace clay
{

class Shape final
{
public:
	//! create a shape with the desired dimensions
	Shape (const std::vector<size_t>& dims);

	//! assign the desired dimensions to this shape
	Shape& operator = (const std::vector<size_t>& dims);

	// >>>> AVOID OVERRIDE <<<<
	Shape (void) = default;

	Shape (const Shape&) = default;

	Shape (Shape&&) = default;

	Shape& operator = (const Shape&) = default;

	Shape& operator = (Shape&&) = default;


	// >>>> ACCESSORS <<<<
	//! access value at index dim, throws std::out_of_range if dim >= rank
	size_t operator [] (size_t dim) const;

	//! get a copy of the shape as a list
	//! accounts for grouping
	std::vector<size_t> as_list (void) const;

	//! get number of elems that can fit in shape if known,
	//! 0 if unknown
	//! accounts for grouping
	size_t n_elems (void) const;

	//! get the minimum number of elements that can fit in shape
	//! return the product of all known dimensions
	//! accounts for grouping
	size_t n_known (void) const;

	//! get shape rank
	size_t rank (void) const;

	//! check if the shape is compatible with other
	//! does not accounts for grouping
	bool is_compatible_with (const Shape& other) const;

	//! check if shape is partially defined
	//! (if there are unknowns but rank is not 0)
	bool is_part_defined (void) const;

	//! check if shape is  fully defined
	//! (there are no unknowns)
	bool is_fully_defined (void) const;

	// >>>> MUTATORS <<<<
	//! invalidate shape
	void undefine (void);

	// >>>> SHAPE CREATORS <<<<
	//! create the most defined shape from this and other
	//! prioritizes this over other value
	Shape merge_with (const Shape& other) const;

	//! create a copy of this shape with leading and
	//! trailing ones removed
	Shape trim (void) const;

	//! create a shape that is the concatenation of another shape
	Shape concatenate (const Shape& other) const;

	//! create a new tensors with same dimension
	//! value and the specified rank
	//! clip or pad with 1's to fit rank
	Shape with_rank (size_t rank) const;

	//! create a new tensors with same dimension
	//! value and at least the the specified rank
	Shape with_rank_at_least (size_t rank) const;

	//! create a new tensors with same dimension
	//! value and at most the the specified rank
	Shape with_rank_at_most (size_t rank) const;

	// todo: test
	// >>>> COORDINATES <<<<
	//! obtain the flat vector index from cartesian coordinates (e.g.: 2-D [x, y] has flat index = y * dimensions_[0] + x)
	size_t flat_idx (std::vector<size_t> coord) const;

	//! obtain cartesian coordinates given a flat vector index
	std::vector<size_t> coord_from_idx (size_t idx) const;

private:
	//! zero values denotes unknown/undefined value
	//! emtpy dimension_ denotes undefined shape
	std::vector<size_t> dimensions_;
};

}

#endif /* CLAY_SHAPE_HPP */
