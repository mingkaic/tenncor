///
/// candidate.hpp
/// opt
///
/// Purpose:
/// Define TEQ graph intermediate representations
///

#include <string>
#include <unordered_map>

#include <boost/functional/hash.hpp>

#include "teq/itensor.hpp"

#ifndef OPT_CAND_HPP
#define OPT_CAND_HPP

namespace opt
{

/// Set of tensors that potentially matches some id
using CtxValT = std::set<teq::TensptrT>;

/// Map of rule graph leaf identifiers to corresponding matches
using ContexT = std::map<std::string,CtxValT>;

/// Set of contexts that serve as a candidates of a conversion rule
using CtxsT = std::unordered_set<ContexT,boost::hash<ContexT>>;

/// Conversion type
enum CAND_TYPE
{
	/// Convert to a scalar
	SCALAR = 0,
	/// Convert to a non-scalar constant
	CONST,
	/// Intermediate conversion
	INTERM,
	/// Full conversion to a subgraph
	CONVRT,
};

/// Generic representation of a conversion rule
struct Symbol final
{
	/// Type of rule
	CAND_TYPE type_;

	// type_=SCALAR: scalar label
	// type_=INTERM: intermediate id
	// type_=CONVRT: conversion ref
	std::string reference_;
};

/// Hasher to encode rule key
struct SymbolHash final
{
	/// Return hash of Symbol
	size_t operator() (const Symbol& sym) const
	{
		size_t seed = 0;
		boost::hash_combine(seed, sym.type_);
		boost::hash_combine(seed, sym.reference_);
		return seed;
	}
};

/// Compare equality of Symbols
inline bool operator == (const Symbol& lhs, const Symbol& rhs)
{
	return lhs.type_ == rhs.type_ && lhs.reference_ == rhs.reference_;
}

/// Map of convers symbols to its potential candidate conversion rules
using CandsT = std::unordered_map<Symbol,CtxsT,SymbolHash>;

/// Encapsulation of match output argument
struct CandArg
{
	/// Real tensor of the argument
	teq::TensptrT tensor_;

	/// Potential rules contexts that can match subgraph of tensor_
	CandsT candidates_;

	/// Real shaper in the argument
	std::string shaper_;

	/// Real coorder in the argument
	teq::CvrtptrT coorder_;
};

/// Vector of candidate arguments
using CandArgsT = std::vector<CandArg>;

}

#endif // OPT_CAND_HPP
