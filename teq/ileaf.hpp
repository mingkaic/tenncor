///
/// ileaf.hpp
/// teq
///
/// Purpose:
/// Define leafs for tensor equation graph
///

#include "teq/itensor.hpp"
#include "teq/idata.hpp"

#ifndef TEQ_ILEAF_HPP
#define TEQ_ILEAF_HPP

namespace teq
{

/// Leaf of the graph commonly representing the variable in an equation
struct iLeaf : public iTensor, public iData
{
	virtual ~iLeaf (void) = default;

	/// Implementation of iTensor
	void accept (iTraveler& visiter) override
	{
		visiter.visit(this);
	}

	/// Return true if leaf is immutable, otherwise false
	virtual bool is_const (void) const = 0;
};

/// Leaf smart pointer
using LeafptrT = std::shared_ptr<iLeaf>;

static const size_t label_limit = 5;

/// Return constant data string representation
template <typename T>
std::string const_encode (const T* data, const teq::Shape& shape)
{
	size_t nelems = shape.n_elems();
	if (std::all_of(data + 1, data + nelems,
		[&](const T& e) { return e == data[0]; }))
	{
		if (0 == data[0]) // prevent -0
		{
			return "0";
		}
		return fmts::to_string(data[0]);
	}
	auto out = fmts::to_string(data,
		data + std::min(label_limit, nelems));
	if (nelems > label_limit)
	{
		out += "...";
	}
	return out;
}

}

#endif // TEQ_ILEAF_HPP
