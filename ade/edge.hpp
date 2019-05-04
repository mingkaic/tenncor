#include "ade/ifunctor.hpp"

#ifndef ADE_EDGE_HPP
#define ADE_EDGE_HPP

namespace ade
{

struct Edge final
{
	bool expired (void) const
	{
		return parent_.expired() || child_.expired();
	}

	TensrefT parent_;

	TensrefT child_;

	Opcode edge_code_;
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

#endif // ADE_EDGE_HPP
