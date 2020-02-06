
#ifndef EXPERIMENTAL_OPT_TARGET_HPP
#define EXPERIMENTAL_OPT_TARGET_HPP

#include "teq/logs.hpp"

#include "query/query.hpp"

#include "experimental/opt/optimize.pb.h"

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

	virtual TargptrT make_symbol (std::string symbol) const = 0;

	virtual TargptrT make_functor (std::string opname,
		const google::protobuf::Map<std::string,query::Attribute>& attrs,
		const TargptrsT& args) const = 0;
};

TargptrT parse_target (const TargetNode& root,
	const iTargetFactory& builder);

}

#endif // EXPERIMENTAL_OPT_TARGET_HPP
