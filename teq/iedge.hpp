///
/// iedge.hpp
/// teq
///
/// Purpose:
/// Define functor argument wrapper to carryover shape and coordinate mappers
///

#include "marsh/objs.hpp"

#include "teq/itensor.hpp"

#ifndef TEQ_IEDGE_HPP
#define TEQ_IEDGE_HPP

namespace teq
{

/// Interface to represent parent-child tensor relationship
struct iEdge
{
	virtual ~iEdge (void) = default;

	/// Return argument tensor shape
	virtual Shape shape (void) const = 0;

	/// Return argument tensor
	virtual TensptrT get_tensor (void) const = 0;

	// todo: deprecate edge attrs
	/// Set attribute in out object specified by attr
	/// If attr is empty string, set all attributes to out marsh::Maps
	virtual void get_attrs (marsh::Maps& out) const = 0;
};

using CEdgesT = std::vector<std::reference_wrapper<const iEdge>>;

}

#endif // TEQ_IEDGE_HPP
