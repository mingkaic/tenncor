//
//  tshape.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/tensor/tshape.hpp"

#ifdef TENNCOR_TENSORSHAPE_HPP

namespace nnet
{

tshape::tshape (const std::vector<size_t>& dims) :
	dimensions_(dims) {}

tshape& tshape::operator = (const std::vector<size_t>& dims)
{
	dimensions_ = dims;
	return *this;
}

size_t tshape::operator [] (size_t dim) const
{
	return dimensions_.at(dim);
}

std::vector<size_t> tshape::as_list (void) const
{
	return dimensions_;
}

size_t tshape::n_elems (void) const
{
	if (dimensions_.empty())
	{
		return 0;
	}
	size_t elems = std::accumulate(dimensions_.begin(), dimensions_.end(),
	(size_t) 1, std::multiplies<size_t>());
	return elems;
}

size_t tshape::n_known (void) const
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

size_t tshape::rank (void) const
{
	return dimensions_.size();
}

bool tshape::is_compatible_with (const tshape& other) const
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

bool tshape::is_part_defined (void) const
{
	return !dimensions_.empty();
}

bool tshape::is_fully_defined (void) const
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

void tshape::undefine (void) { dimensions_.clear(); }

tshape tshape::merge_with (const tshape& other) const
{
	if (dimensions_.empty())
	{
		return other;
	}
	if (other.dimensions_.empty())
	{
		return *this;
	}
	if (dimensions_.size() != other.dimensions_.size())
	{
		throw std::logic_error(nnutils::formatter() << "shape of rank "
			<< dimensions_.size() << " is not compatible with shape of rank "
			<< other.dimensions_.size());
	}
	std::vector<size_t> ds;
	for (size_t i = 0; i < dimensions_.size(); i++)
	{
		size_t value = dimensions_[i];
		size_t ovalue = other.dimensions_[i];
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

tshape tshape::trim (void) const
{
	std::vector<size_t> res;
	if (false == dimensions_.empty())
	{
		size_t start = 0;
		size_t end = dimensions_.size() - 1;
		while (start < end && 1 == dimensions_.at(start)) { start++; }
		while (start < end && 1 == dimensions_.at(end)) { end--; }
		if (start < end || 1 != dimensions_.at(end))
		{
			res.insert(res.end(),
				dimensions_.begin()+start,
				dimensions_.begin()+end+1);
		}
	}
	return res;
}

tshape tshape::concatenate (const tshape& other) const
{
	if (dimensions_.empty())
	{
		return other;
	}
	if (other.dimensions_.empty())
	{
		return *this;
	}
	std::vector<size_t> ds = dimensions_;
	ds.insert(ds.end(), other.dimensions_.begin(), other.dimensions_.end());
	return tshape(ds);
}

tshape tshape::with_rank (size_t rank) const
{
	size_t ndim = dimensions_.size();
	std::vector<size_t> ds;
	if (rank < ndim)
	{
		// clip to rank
		auto it = dimensions_.begin();
		ds.insert(ds.end(), it, it+rank);
	}
	else if (rank> ndim)
	{
		// pad to fit rank
		ds = dimensions_;
		size_t diff = rank - ndim;
		ds.insert(ds.end(), diff, 1);
	}
	else
	{
		ds = dimensions_;
	}
	return ds;
}

tshape tshape::with_rank_at_least (size_t rank) const
{
	size_t ndim = dimensions_.size();
	std::vector<size_t> ds = dimensions_;
	if (rank> ndim)
	{
		// pad to fit rank
		size_t diff = rank - ndim;
		ds.insert(ds.end(), diff, 1);
	}
	return ds;
}

tshape tshape::with_rank_at_most (size_t rank) const
{
	std::vector<size_t> ds;
	if (rank < dimensions_.size())
	{
		// clip to fit rank
		auto it = dimensions_.begin();
		ds.insert(ds.end(), it, it+rank);
	}
	else
	{
		ds = dimensions_;
	}
	return ds;
}

size_t tshape::flat_idx (std::vector<size_t> coord) const
{
	size_t n = std::min(dimensions_.size(), coord.size());
	size_t index = 0;
	for (size_t i = 1; i < n; i++)
	{
		index += coord[n-i];
		index *= dimensions_[n-i-1];
	}
	return index + coord[0];
}

std::vector<size_t> tshape::coord_from_idx (size_t idx) const
{
	std::vector<size_t> coord;
	size_t xd;
	for (size_t d : dimensions_)
	{
		xd = idx % d;
		coord.push_back(xd);
		idx = (idx - xd) / d;
	}
	return coord;
}

void print_shape (tshape ts, std::ostream& os)
{
	std::vector<size_t> shape = ts.as_list();
	if (shape.empty())
	{
		os << "undefined ";
	}
	else
	{
		for (size_t dim : shape)
		{
			os << dim << " ";
		}
	}
}

}

#endif
