#include "teq/coord.hpp"

#ifndef ETEQ_COORD_HPP
#define ETEQ_COORD_HPP

namespace eteq
{

struct CoordMap final : public teq::iCoordMap
{
	CoordMap (teq::CoordT indices, bool bijective) :
		indices_(indices), bijective_(bijective) {}

	teq::iCoordMap* connect (const teq::iCoordMap& rhs) const override
	{
		return nullptr;
	}

	void forward (teq::CoordT::iterator out,
		teq::CoordT::const_iterator in) const override
	{
		std::copy(indices_.begin(), indices_.end(), out);
	}

	iCoordMap* reverse (void) const override
	{
		return nullptr;
	}

	std::string to_string (void) const override
	{
		return fmts::to_string(indices_.begin(), indices_.end());
	}

	void access (std::function<void(const teq::MatrixT&)> cb) const override {}

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

CoordptrT reduce (std::vector<teq::RankT> red_dims);

CoordptrT extend (teq::RankT rank, std::vector<teq::DimT> ext);

CoordptrT permute (std::vector<teq::RankT> dims);

}

#endif // ETEQ_COORD_HPP
