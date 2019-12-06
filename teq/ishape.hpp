///
/// ishape.hpp
/// teq
///
/// Purpose:
/// Define shape signature that supports unknown dimensions
///

#include <array>
#include <cmath>
#include <numeric>
#include <vector>

#include "logs/logs.hpp"

#ifndef TEQ_ISHAPE_HPP
#define TEQ_ISHAPE_HPP

namespace teq
{

/// Type used for shape rank
using RankT = uint8_t;

/// Type used for shape dimension
#if !defined(SDIM_BYTES) || SDIM_BYTES <= 1
using DimT = uint8_t;
#elif SDIM_BYTES <= 2
using DimT = uint16_t;
#elif SDIM_BYTES <= 4
using DimT = uint32_t;
#else
using DimT = uint64_t;
#endif

/// Type used for flattened index
/// DimT having 8 bits and shape comprising of 8 DimT values means a maximum
/// flattened index of (2 ^ 8) ^ 8 = 2 ^ 64
using NElemT = uint64_t;

/// Number of dimsensions in a shape/coordinate
const RankT rank_cap = 8;

/// Array type used to hold dimension info in Shape
using ShapeT = std::array<DimT,rank_cap>;

/// Type of iterator used to iterate through internal array
using siterator = ShapeT::iterator;

/// Type of constant iterator used to iterate through internal array
using const_siterator = ShapeT::const_iterator;

struct ShapeSignature
{
	ShapeSignature (void)
	{
		std::fill(begin(), end(), 0);
	}

	ShapeSignature (std::vector<DimT> dims)
	{
		vector_assign(dims);
	}

	virtual ~ShapeSignature (void) = default;

	/// Return DimT element at idx for any index in range [0:rank_cap)
	DimT at (RankT idx) const
	{
		if (rank_cap <= idx)
		{
			logs::fatalf("cannot access out of bounds index %d", idx);
		}
		return dims_.at(idx);
	}

	/// Return true if this[0:idx) is equal to other[0:idx),
	/// otherwise return false
	bool compatible_before (const ShapeSignature& other, RankT idx) const
	{
		auto it = dims_.begin();
		return std::equal(it, it + std::min(idx, rank_cap), other.begin(),
			[](DimT a, DimT b) { return a == 0 || b == 0 || a == b; });
	}

	/// Return true if this[idx:rank_cap) is
	/// equal to other[idx:rank_cap), otherwise return false
	/// Set idx to 0 to compare entire shape
	bool compatible_after (const ShapeSignature& other, RankT idx) const
	{
		return idx < rank_cap && std::equal(
			dims_.begin() + idx, dims_.end(), other.dims_.begin() + idx,
			[](DimT a, DimT b) { return a == 0 || b == 0 || a == b; });
	}

	/// Return string representation of shape
	std::string to_string (void) const
	{
		return fmts::to_string(begin(), end());
	}

	// >>>> INTERNAL CONTROL <<<<

	/// Return begin iterator of internal array
	siterator begin (void)
	{
		return dims_.begin();
	}

	/// Return end iterator of internal array
	siterator end (void)
	{
		return dims_.end();
	}

	/// Return begin constant iterator of internal array
	const_siterator begin (void) const
	{
		return dims_.begin();
	}

	/// Return end constant iterator of internal array
	const_siterator end (void) const
	{
		return dims_.end();
	}

protected:
	void vector_assign (const std::vector<DimT>& dims)
	{
		RankT rank = std::min((size_t) rank_cap, dims.size());
		auto src = dims.begin();
		auto dest = this->begin();
		std::copy(src, src + rank, dest);
		std::fill(dest + rank, dest + rank_cap, 1);
	}

	/// Array of dimension values
	ShapeT dims_;
};

}

#endif // TEQ_ISHAPE_HPP
