//
//  shape.cpp
//  clay
//

#include <functional>
#include <numeric>

#include "clay/shape.hpp"
#include "clay/error.hpp"

#include "ioutil/stream.hpp"

#ifdef CLAY_SHAPE_HPP

namespace clay
{

Shape::Shape (const std::vector<size_t>& dims) :
	dimensions_(dims) {}

Shape& Shape::operator = (const std::vector<size_t>& dims)
{
	dimensions_ = dims;
	return *this;
}

size_t Shape::at (size_t dim) const
{
	return dimensions_.at(dim);
}

typename std::vector<size_t>::const_iterator Shape::begin (void) const
{
	return dimensions_.cbegin();
}

typename std::vector<size_t>::const_iterator Shape::end (void) const
{
	return dimensions_.cend();
}

std::vector<size_t> Shape::as_list (void) const
{
	return dimensions_;
}

size_t Shape::n_elems (void) const
{
	if (dimensions_.empty())
	{
		return 0;
	}
	size_t elems = std::accumulate(dimensions_.begin(), dimensions_.end(),
	(size_t) 1, std::multiplies<size_t>());
	return elems;
}

size_t Shape::n_known (void) const
{
	if (dimensions_.empty())
	{
		return 0;
	}
	size_t elems = std::accumulate(dimensions_.begin(), dimensions_.end(),
	(size_t) 1,
	[](size_t a, size_t b) {
		if (b != 0)
		{
			return a * b;
		}
		return a;
	});
	return elems;
}

size_t Shape::rank (void) const
{
	return dimensions_.size();
}

bool Shape::is_compatible_with (const Shape& other) const
{
	bool incomp = true;
	if (!dimensions_.empty() && !other.dimensions_.empty())
	{
		size_t thisn = dimensions_.size();
		size_t othern = other.dimensions_.size();
		size_t beginthis = 0;
		size_t beginother = 0;
		// invariant thisn and othern>= 1 (since dimensions are not empty)
		size_t endthis = thisn-1;
		size_t endother = othern-1;

		if (thisn != othern)
		{
			while (beginthis < thisn-1 && 1 == dimensions_[beginthis]) { beginthis++; }
			while (endthis> beginthis && 1 == dimensions_[endthis]) { endthis--; }
			while (beginother < othern-1 && 1 == other.dimensions_[beginother]) { beginother++; }
			while (endother> beginother && 1 == other.dimensions_[endother]) { endother--; }
			size_t lenthis = endthis - beginthis;
			size_t lenother = endother - beginother;
			if (lenthis> lenother)
			{
				// todo: improve this matching algorithm to account for cases where
				// decrementing endthis before incrementing beginthis matches while the opposite order doesn't

				// try to match this to other by searching for padding zeros to convert to 1 padding in this
				while (endthis - beginthis> lenother && beginthis < endthis && 0 == dimensions_[beginthis]) { beginthis++; }
				while (endthis - beginthis> lenother && endthis> beginthis && 0 == dimensions_[endthis]) { endthis--; }

				if (endthis - beginthis> lenother)
					// match unsuccessful, they are incompatible
					return false;
			}
			else if (lenother> lenthis)
			{
				// try to match other to this by searching for padding zeros to convert to 1 padding in other
				while (endother - beginother> lenthis && beginother < endother && 0 == other.dimensions_[beginother]) { beginother++; }
				while (endother - beginother> lenthis && endother> beginother && 0 == other.dimensions_[endother]) { endother--; }

				if (endother - beginother> lenthis)
					// match unsuccessful, they are incompatible
					return false;
			}
		}

		// invariant: endthis - beginthis == endother - beginother
		while (beginthis <= endthis && beginother <= endother)
		{
			incomp = incomp &&
				(dimensions_[beginthis] == other.dimensions_[beginother] ||
				0 == (dimensions_[beginthis] && other.dimensions_[beginother]));
			beginthis++;
			beginother++;
		}
	}
	return incomp;
}

bool Shape::is_part_defined (void) const
{
	return !dimensions_.empty();
}

bool Shape::is_fully_defined (void) const
{
	if (dimensions_.empty())
	{
		return false;
	}
	bool known = true;
	for (size_t d : dimensions_)
	{
		known = known && (0 < d);
	}
	return known;
}

Shape merge_with (const Shape& shape, const Shape& other)
{
	if (false == shape.is_part_defined())
	{
		return other;
	}
	if (false == other.is_part_defined())
	{
		return shape;
	}
	size_t rank = shape.rank();
	if (rank != other.rank())
	{
		throw InvalidShapeError(shape, other);
	}
	std::vector<size_t> ds;
	for (size_t i = 0; i < rank; i++)
	{
		size_t value = shape.at(i);
		size_t ovalue = other.at(i);
		if (value == ovalue || (value && ovalue))
		{
			ds.push_back(value);
		}
		else
		{
			// one of the values is zero, return the non-zero value
			ds.push_back(value + ovalue);
		}
	}
	return ds;
}

Shape trim (const Shape& shape)
{
	std::vector<size_t> res;
	if (shape.is_part_defined())
	{
		size_t start = 0;
		size_t end = shape.rank() - 1;
		while (start < end && 1 == shape.at(start)) { start++; }
		while (start < end && 1 == shape.at(end)) { end--; }
		if (start < end || 1 != shape.at(end))
		{
			res.insert(res.end(), shape.begin() + start,
				shape.begin() + end + 1);
		}
	}
	return res;
}

Shape concatenate (const Shape& shape, const Shape& other)
{
	if (false == shape.is_part_defined())
	{
		return other;
	}
	if (false == other.is_part_defined())
	{
		return shape;
	}
	std::vector<size_t> ds = shape.as_list();
	ds.insert(ds.end(), other.begin(), other.end());
	return Shape(ds);
}

Shape with_rank (const Shape& shape, size_t rank)
{
	size_t ndim = shape.rank();
	std::vector<size_t> ds;
	if (rank < ndim)
	{
		// clip to rank
		auto it = shape.begin();
		ds.insert(ds.end(), it, it+rank);
	}
	else if (rank> ndim)
	{
		// pad to fit rank
		ds = shape.as_list();
		size_t diff = rank - ndim;
		ds.insert(ds.end(), diff, 1);
	}
	else
	{
		ds = shape.as_list();
	}
	return ds;
}

Shape with_rank_at_least (const Shape& shape, size_t rank)
{
	size_t ndim = shape.rank();
	std::vector<size_t> ds = shape.as_list();
	if (rank> ndim)
	{
		// pad to fit rank
		size_t diff = rank - ndim;
		ds.insert(ds.end(), diff, 1);
	}
	return ds;
}

Shape with_rank_at_most (const Shape& shape, size_t rank)
{
	std::vector<size_t> ds;
	if (rank < shape.rank())
	{
		// clip to fit rank
		auto it = shape.begin();
		ds.insert(ds.end(), it, it+rank);
	}
	else
	{
		ds = shape.as_list();
	}
	return ds;
}

size_t index (const Shape& shape, std::vector<size_t> coord)
{
	size_t n = std::min(shape.rank(), coord.size());
	size_t index = 0;
	for (size_t i = 1; i < n; i++)
	{
		index += coord[n - i];
		index *= shape.at(n - i - 1);
	}
	return index + coord[0];
}

std::vector<size_t> coordinate (const Shape& shape, size_t idx)
{
	std::vector<size_t> coord;
	size_t xd;
	for (auto it = shape.begin(); it != shape.end(); ++it)
	{
		xd = idx % *it;
		coord.push_back(xd);
		idx = (idx - xd) / *it;
	}
	return coord;
}

std::string to_string (const Shape& shape)
{
	return ioutil::Stream() << shape.as_list();
}

}

#endif
