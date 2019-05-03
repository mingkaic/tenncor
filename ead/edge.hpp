#include "ade/itensor.hpp"
#include "ade/coord.hpp"
#include "ade/ifunctor.hpp"

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

struct EdgeHash final
{
	size_t operator() (const Edge& edge) const
	{
		if (edge.expired())
		{
			return 0;
		}
		std::stringstream ss;
		ss << edge.parent_.lock().get() <<
			edge.child_.lock().get() <<
			edge.edge_code_.code_;
		return std::hash<std::string>()(ss.str());
	}
};

inline bool operator == (const Edge& lhs, const Edge& rhs)
{
	EdgeHash hasher;
    return hasher(lhs) == hasher(rhs);
}

}

#endif // EAD_EDGE_HPP
