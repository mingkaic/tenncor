#include <array>
#include <vector>

#ifndef SHAPE_HPP
#define SHAPE_HPP

using DimT = uint8_t;
using NElemT = uint32_t; // reliant on Shape::rank_limit_
const uint8_t rank_cap = 8;

// aligned shape representation
struct Shape
{
	using ShapeIterator = std::array<DimT,rank_cap>::iterator;

	Shape (void);

	Shape (std::vector<DimT> dims);

	Shape (std::vector<Shape> shapes);

	Shape (const Shape& other);

	Shape& operator = (const Shape& other);

	Shape& operator = (const std::vector<DimT>& dims);

	Shape (Shape&& other);

	Shape& operator = (Shape&& other);


	// >>>> ACCESSORS <<<<

	DimT at (uint8_t idx) const;

	Shape group (uint8_t idx) const;

	uint8_t n_rank (void) const;

	NElemT n_elems (void) const;

	std::vector<DimT> as_list (void) const;

	std::vector<Shape> as_groups (void) const;

	bool compatible_before (const Shape& other, uint8_t idx) const;

	//! check if this is compatible
	//! with another shape after a certain rank
	bool compatible_after (const Shape& other, uint8_t idx) const;

	std::string to_string (void) const;

	// >>>> ITERATORS <<<<

	ShapeIterator begin (void);

	ShapeIterator end (void);

private:
	void vector_assign (const std::vector<DimT>& dims);

	void move_helper (Shape&& other);

	// get the first index of the kth group
	uint8_t kth_group (uint8_t k) const
	{
		// [lsb] 0 ... rank_cap-1 [msb]
		uint8_t i = 0;
		for (; i < rank_cap && k > 0; ++i)
		{
			k -= (groups_ >> i) & 1;
		}
		return i;
	}

	std::array<DimT,rank_cap> dims_; // todo: add padding for groups

	uint8_t groups_ = 0xFF;

	uint8_t rank_;
};

bool higher_order (Shape shape);

//! obtain the flat vector index from cartesian coordinates
//! (e.g.: 2-D [x, y] has flat index = y * dimensions_[0] + x)
NElemT index (Shape shape, std::vector<DimT> coord);

//! obtain cartesian coordinates given a flat vector index
std::vector<DimT> coordinate (Shape shape, NElemT idx);

#endif /* SHAPE_HPP */
