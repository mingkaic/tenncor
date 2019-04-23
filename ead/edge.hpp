#include "ade/itensor.hpp"
#include "ade/coord.hpp"

#ifndef EAD_EDGE_HPP
#define EAD_EDGE_HPP

namespace ead
{

enum EDGE_CODE
{
	GRADIENT = 0,
};

struct Edge final
{
	bool expired (void) const
	{
		return parent_.expired() || child_.expired();
	}

	std::weak_ptr<ade::iTensor> parent_;

	std::weak_ptr<ade::iTensor> child_;

	ade::Opcode edge_code_;
};

using EdgesT = std::vector<Edge>;

}

#endif // EAD_EDGE_HPP
