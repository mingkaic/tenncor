///
/// iedge.hpp
/// teq
///
/// Purpose:
/// Define functor argument wrapper to carryover shape and coordinate mappers
///

#include "marsh/attrs.hpp"

#include "teq/itensor.hpp"

#ifndef TEQ_IEDGE_HPP
#define TEQ_IEDGE_HPP

namespace teq
{

/// Interface to represent parent-child tensor relationship
struct iEdge : public marsh::iAttributed
{
	virtual ~iEdge (void) = default;

	/// Return argument tensor shape
	virtual Shape shape (void) const = 0;

	/// Return argument tensor
	virtual TensptrT get_tensor (void) const = 0;
};

using EdgeSetT = std::unordered_set<iEdge*>;

using EdgeptrT = std::shared_ptr<iEdge>;

using EdgeptrsT = std::vector<EdgeptrT>;

using EdgeRefsT = std::vector<std::reference_wrapper<const iEdge>>;

}

#endif // TEQ_IEDGE_HPP
