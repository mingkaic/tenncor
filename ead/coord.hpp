#include "ade/coord.hpp"

#ifndef EAD_COORD_HPP
#define EAD_COORD_HPP

namespace ead
{

enum TransCode
{
	EXTEND = 0,
	PERMUTE,
	REDUCE,
};

struct CoordMap final : public ade::iCoordMap
{
	CoordMap (TransCode transcode, ade::CoordT indices, bool bijective) :
		transcode_(transcode), indices_(indices), bijective_(bijective) {}

	ade::iCoordMap* connect (const ade::iCoordMap& rhs) const override
	{
		return nullptr; // todo: implement
	}

	void forward (ade::CoordT::iterator out,
		ade::CoordT::const_iterator in) const override
	{
		std::copy(indices_.begin(), indices_.end(), out);
	}

	iCoordMap* reverse (void) const override
	{
		return nullptr; // todo: implement
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

	TransCode transcode (void) const
	{
		return transcode_;
	}

private:
	TransCode transcode_;

	ade::CoordT indices_;

	bool bijective_;
};

/// Type of iCoordMap smartpointer
using CoordptrT = std::shared_ptr<CoordMap>;

CoordptrT reduce (std::vector<uint8_t> red_dims);

CoordptrT extend (uint8_t rank, std::vector<ade::DimT> ext);

CoordptrT permute (std::vector<uint8_t> dims);

}

#endif // EAD_COORD_HPP
