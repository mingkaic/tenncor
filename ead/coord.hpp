#include "ade/coord.hpp"

#ifndef EAD_COORD_HPP
#define EAD_COORD_HPP

namespace ead
{

struct CoordMap final : public ade::iCoordMap
{
	CoordMap (ade::CoordT indices, bool bijective) :
		indices_(indices), bijective_(bijective) {}

	ade::iCoordMap* connect (const ade::iCoordMap& rhs) const override
	{
		return nullptr;
	}

	void forward (ade::CoordT::iterator out,
		ade::CoordT::const_iterator in) const override
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

	void access (std::function<void(const ade::MatrixT&)> cb) const override {}

	bool is_bijective (void) const override
	{
		return bijective_;
	}

private:
	ade::CoordT indices_;

	bool bijective_;
};

/// Type of iCoordMap smartpointer
using CoordptrT = std::shared_ptr<CoordMap>;

CoordptrT reduce (std::vector<ade::RankT> red_dims);

CoordptrT extend (ade::RankT rank, std::vector<ade::DimT> ext);

CoordptrT permute (std::vector<ade::RankT> dims);

}

#endif // EAD_COORD_HPP
