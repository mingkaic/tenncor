
#ifndef ETEQ_TYPER_HPP
#define ETEQ_TYPER_HPP

#include "internal/eigen/eigen.hpp"

namespace eteq
{

const std::string nointype_err = "cannot infer type without input types(s)";

using DTypesT = std::vector<egen::_GENERATED_DTYPE>;

template <egen::_GENERATED_OPCODE OPCODE>
struct TypeParser final
{
	/// Return output datatype infered from input dtypes
	/// By default infer the max precision datatype
	egen::_GENERATED_DTYPE dtype (const marsh::iAttributed& attrs,
		const DTypesT& dtypes) const
	{
		if (dtypes.empty())
		{
			global::fatal(nointype_err);
		}
		return *std::max_element(dtypes.begin(), dtypes.end(),
		[](egen::_GENERATED_DTYPE lhs, egen::_GENERATED_DTYPE rhs)
		{
			return egen::type_precision(lhs) < egen::type_precision(rhs);
		});
	}
};

struct AssignTyper
{
	virtual ~AssignTyper (void) = default;

	egen::_GENERATED_DTYPE dtype (const marsh::iAttributed& attrs,
		const DTypesT& dtypes) const
	{
		if (dtypes.empty())
		{
			global::fatal(nointype_err);
		}
		return dtypes.front();
	}
};

template <>
struct TypeParser<egen::ASSIGN> final : private AssignTyper
{
	using AssignTyper::dtype;
};

template <>
struct TypeParser<egen::ASSIGN_ADD> final : private AssignTyper
{
	using AssignTyper::dtype;
};

template <>
struct TypeParser<egen::ASSIGN_SUB> final : private AssignTyper
{
	using AssignTyper::dtype;
};

template <>
struct TypeParser<egen::ASSIGN_MUL> final : private AssignTyper
{
	using AssignTyper::dtype;
};

template <>
struct TypeParser<egen::ASSIGN_DIV> final : private AssignTyper
{
	using AssignTyper::dtype;
};

}

#endif // ETEQ_TYPER_HPP
