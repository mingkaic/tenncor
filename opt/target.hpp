
#ifndef OPT_TARGET_HPP
#define OPT_TARGET_HPP

#include "query/query.hpp"

namespace opt
{

struct iTarget
{
	virtual ~iTarget (void) = default;

	virtual teq::TensptrT convert (
		const query::SymbMapT& candidates) const = 0;
};

using TargptrT = std::shared_ptr<iTarget>;

using TargptrsT = std::vector<TargptrT>;

struct iTargetFactory
{
	virtual ~iTargetFactory (void) = default;

	virtual TargptrT make_scalar (double scalar,
		std::string sshape) const = 0;

	virtual TargptrT make_symbol (const std::string& symbol) const = 0;

	virtual TargptrT make_functor (const std::string& opname,
		const google::protobuf::Map<std::string,query::Attribute>& attrs,
		const TargptrsT& args) const = 0;
};

}

#endif // OPT_TARGET_HPP
