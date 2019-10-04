///
/// coord.hpp
/// eteq
///
/// Purpose:
/// Define Eigen transformation argument wrapper
///

#include "teq/coord.hpp"

#ifndef ETEQ_COORD_HPP
#define ETEQ_COORD_HPP

namespace eteq
{

/// Eigen transformation wrapper implementation of iCoordMap
struct CoordMap final : public teq::iCoordMap
{
	CoordMap (teq::CoordT indices, bool bijective) :
		indices_(indices), bijective_(bijective) {}

	/// Implementation of iCoordMap
	teq::iCoordMap* connect (const teq::iCoordMap& rhs) const override
	{
		return nullptr;
	}

	/// Implementation of iCoordMap
	void forward (teq::CoordT::iterator out,
		teq::CoordT::const_iterator in) const override
	{
		std::copy(indices_.begin(), indices_.end(), out);
	}

	/// Implementation of iCoordMap
	iCoordMap* reverse (void) const override
	{
		return nullptr;
	}

	/// Implementation of iCoordMap
	std::string to_string (void) const override
	{
		return fmts::to_string(indices_.begin(), indices_.end());
	}

	/// Implementation of iCoordMap
	void access (std::function<void(const teq::MatrixT&)> cb) const override {}

	/// Implementation of iCoordMap
	bool is_bijective (void) const override
	{
		return bijective_;
	}

private:
	teq::CoordT indices_;

	bool bijective_;
};

/// Type of iCoordMap smartpointer
using CoordptrT = std::shared_ptr<CoordMap>;

/// Return CoordMap wrapper of reduction dimensions
CoordptrT reduce (std::vector<teq::RankT> red_dims);

/// Return CoordMap wrapper of extension parameters
CoordptrT extend (teq::RankT rank, std::vector<teq::DimT> ext);

/// Return CoordMap wrapper of permute indices
CoordptrT permute (std::vector<teq::RankT> dims);

}

#endif // ETEQ_COORD_HPP
