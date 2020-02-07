
#ifndef EXPERIMENTAL_ETEQ_OPT_TARGET_HPP
#define EXPERIMENTAL_ETEQ_OPT_TARGET_HPP

#include "experimental/opt/parse.hpp"

#include "eteq/make.hpp"

namespace eteq
{

template <typename T>
struct ScalarTarget final : public opt::iTarget
{
	ScalarTarget (double scalar, const std::string& sshape) :
		scalar_(scalar), symb_(sshape) {}

	teq::TensptrT convert (const query::SymbMapT& candidates) const override
	{
		return eteq::make_constant_scalar(scalar_,
			candidates.at(symb_)->shape());
	}

	double scalar_;

	std::string symb_;
};


template <typename T>
struct SymbolTarget final : public opt::iTarget
{
	SymbolTarget (const std::string& symb, const opt::GraphInfo& graph) :
		symb_(symb), graph_(&graph) {}

	teq::TensptrT convert (const query::SymbMapT& candidates) const override
	{
		return graph_->owners_.at(candidates.at(symb_)).lock();
	}

	std::string symb_;

	const opt::GraphInfo* graph_;
};


template <typename T>
struct FunctorTarget final : public opt::iTarget
{
	FunctorTarget (const std::string& opname, const opt::TargptrsT& args,
		marsh::Maps&& attr) : opname_(opname), targs_(args),
		attr_(std::move(attr)) {}

	teq::TensptrT convert (const query::SymbMapT& candidates) const override
	{
		marsh::Maps* attrcpy = attr_.clone();
		teq::TensptrsT args;
		args.reserve(targs_.size());
		for (opt::TargptrT target : targs_)
		{
			args.push_back(target->convert(candidates));
		}
		return eteq::make_funcattr<T>(egen::get_op(opname_), args, *attrcpy);
	}
	// testcase:
	// reduce_sum(scalar)
	// extend(scalar)
	// mul(scalar, scalar2)
	// matmul(scalar, scalar2)
	// add(shape([2, 3]), scalar2)
	// add(scalar, shape([2, 3]))
	// matmul(shape([2, 3]), scalar2)
	// matmul(shape([2, 3]), shape([4, 2]))

	std::string opname_;

	opt::TargptrsT targs_;

	marsh::Maps attr_;
};


template <typename T>
struct TargetFactory final : public opt::iTargetFactory
{
	TargetFactory (const opt::GraphInfo& graphinfo) :
		ginfo_(&graphinfo) {}

	opt::TargptrT make_scalar (double scalar,
		std::string sshape) const override
	{
		return std::make_shared<ScalarTarget<T>>(scalar, sshape);
	}

	opt::TargptrT make_symbol (std::string symbol) const override
	{
		return std::make_shared<SymbolTarget<T>>(symbol, *ginfo_);
	}

	opt::TargptrT make_functor (std::string opname,
		const google::protobuf::Map<std::string,query::Attribute>& attrs,
		const opt::TargptrsT& args) const override
	{
		marsh::Maps attrmap;
		for (const auto& attrpair : attrs)
		{
			attrmap.add_attr(attrpair.first,
				marsh::ObjptrT(opt::parse(attrpair.second, *ginfo_)));
		}
		return std::make_shared<FunctorTarget<T>>(opname, args, std::move(attrmap));
	}

	const opt::GraphInfo* ginfo_;
};

}

#endif // EXPERIMENTAL_ETEQ_OPT_TARGET_HPP
