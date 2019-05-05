///
/// ead.hpp
/// ade
///
/// Purpose:
/// Define non-operating edges between tensor nodes
///

#include "ade/ifunctor.hpp"

#ifndef ADE_EDGE_HPP
#define ADE_EDGE_HPP

namespace ade
{

/// Edge between parent and child tensor references and labelled with edge code
struct Edge final
{
	/// Return true if either parent or child are expired, otherwise false
	bool expired (void) const
	{
		return parent_.expired() || child_.expired();
	}

	/// Parent reference
	TensrefT parent_;

	/// Child refence
	TensrefT child_;

	/// Edge label and enumeration
	Opcode edge_code_;
};

/// Vector of edges
using EdgesT = std::vector<Edge>;

/// Edge hasher
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

/// Edge equality comparator
inline bool operator == (const Edge& lhs, const Edge& rhs)
{
	EdgeHash hasher;
    return hasher(lhs) == hasher(rhs);
}

}

#endif // ADE_EDGE_HPP
