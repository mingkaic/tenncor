///
/// ileaf.hpp
/// teq
///
/// Purpose:
/// Define leafs for tensor equation graph
///

#ifndef TEQ_ILEAF_HPP
#define TEQ_ILEAF_HPP

#include "internal/teq/itensor.hpp"

namespace teq
{

enum Usage
{
	UNKNOWN_USAGE = 0,
	IMMUTABLE,
	VARUSAGE,
	PLACEHOLDER,
};

/// Leaf of the graph commonly representing the variable in an equation
struct iLeaf : public iTensor
{
	virtual ~iLeaf (void) = default;

	iLeaf* clone (void) const
	{
		return static_cast<iLeaf*>(this->clone_impl());
	}

	/// Implementation of iTensor
	void accept (iTraveler& visiter) override
	{
		visiter.visit(*this);
	}

	/// Return intended usage context
	virtual Usage get_usage (void) const = 0;
};

/// Leaf smart pointer
using LeafptrT = std::shared_ptr<iLeaf>;

using LeafsT = std::vector<iLeaf*>;

template <typename T>
using LeafMapT = std::unordered_map<iLeaf*,T>;

static const size_t label_limit = 5;

/// Return constant data string representation
template <typename T>
std::string const_encode (const T* data, const Shape& shape)
{
	size_t nelems = shape.n_elems();
	if (std::all_of(data + 1, data + nelems,
		[&](const T& e) { return e == data[0]; }))
	{
		return fmts::to_string(data[0]);
	}
	std::string out;
	if (nelems > label_limit)
	{
		types::StringsT strs;
		strs.reserve(label_limit + 1);
		std::transform(data, data + label_limit, std::back_inserter(strs),
			[](T e) { return fmts::to_string(e); });
		strs.push_back("...");
		out = fmts::to_string(strs.begin(), strs.end());
	}
	else
	{
		out = fmts::to_string(data, data + std::min(label_limit, nelems));
	}
	return out;
}

Usage get_named_usage (const std::string& name);

std::string get_usage_name (Usage usage);

}

#endif // TEQ_ILEAF_HPP
