#include <array>
#include <vector>

#ifndef SAND_SHAPE_HPP
#define SAND_SHAPE_HPP

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

	Shape (std::vector<DimT> dims, uint8_t group) :
		Shape(dims)
	{
		groups_ = group;
	}

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

	uint8_t groups_encoding (void) const
	{
		return groups_;
	}

	// >>>> ITERATORS <<<<

	ShapeIterator begin (void);

	ShapeIterator end (void);

private:
	void vector_assign (const std::vector<DimT>& dims);

	void move_helper (Shape&& other);

	// get the first index of the kth group
	uint8_t kth_group (uint8_t k) const;

	std::array<DimT,rank_cap> dims_;

	uint8_t groups_ = 0xFF; // todo: deprecate by opstring system

	uint8_t rank_;
};

bool higher_order (Shape shape);

//! obtain the flat vector index from cartesian coordinates
//! (e.g.: 2-D [x, y] has flat index = y * dimensions_[0] + x)
NElemT index (Shape shape, std::vector<DimT> coord);

//! obtain cartesian coordinates given a flat vector index
std::vector<DimT> coordinate (Shape shape, NElemT idx);

#endif /* SAND_SHAPE_HPP */
