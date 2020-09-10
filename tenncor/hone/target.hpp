
#ifndef HONE_TARGET_HPP
#define HONE_TARGET_HPP

#include "internal/opt/opt.hpp"

#include "tenncor/eteq/eteq.hpp"

namespace hone
{

#define _CHOOSE_SCALAR_TARGETTYPE(REALTYPE)\
out = eteq::make_constant_scalar<REALTYPE>((REALTYPE)scalar_, outshape);

struct ScalarTarget final : public opt::iTarget
{
	ScalarTarget (double scalar, const std::string& sshape) :
		scalar_(scalar), symb_(sshape) {}

	teq::TensptrT convert (
		const query::SymbMapT& candidates) const override
	{
		auto cand = candidates.at(symb_);
		teq::TensptrT out;
		auto outtype = (egen::_GENERATED_DTYPE) cand->get_meta().type_code();
		auto outshape = cand->shape();
		TYPE_LOOKUP(_CHOOSE_SCALAR_TARGETTYPE, outtype);
		return out;
	}

	double scalar_;

	std::string symb_;
};

#undef _CHOOSE_SCALAR_TARGETTYPE

struct SymbolTarget final : public opt::iTarget
{
	SymbolTarget (const std::string& symb, const opt::GraphInfo& graph) :
		symb_(symb), graph_(&graph) {}

	teq::TensptrT convert (
		const query::SymbMapT& candidates) const override
	{
		return graph_->get_owner(candidates.at(symb_));
	}

	std::string symb_;

	const opt::GraphInfo* graph_;
};

struct FunctorTarget final : public opt::iTarget
{
	FunctorTarget (const std::string& opname, const opt::TargptrsT& args,
		marsh::Maps&& attr) : opname_(opname), targs_(args),
		attr_(std::move(attr)) {}

	teq::TensptrT convert (const query::SymbMapT& candidates) const override
	{
		std::unique_ptr<marsh::Maps> attrcpy(attr_.clone());
		teq::TensptrsT args;
		args.reserve(targs_.size());
		for (opt::TargptrT target : targs_)
		{
			args.push_back(target->convert(candidates));
		}
		return eteq::make_funcattr(egen::get_op(opname_), args, *attrcpy);
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

struct TargetFactory final : public opt::iTargetFactory
{
	TargetFactory (const opt::GraphInfo& graphinfo) :
		ginfo_(&graphinfo) {}

	opt::TargptrT make_scalar (double scalar,
		std::string sshape) const override
	{
		return std::make_shared<ScalarTarget>(scalar, sshape);
	}

	opt::TargptrT make_symbol (const std::string& symbol) const override
	{
		return std::make_shared<SymbolTarget>(symbol, *ginfo_);
	}

	opt::TargptrT make_functor (const std::string& opname,
		const google::protobuf::Map<std::string,query::Attribute>& attrs,
		const opt::TargptrsT& args) const override
	{
		marsh::Maps attrmap;
		for (const auto& attrpair : attrs)
		{
			attrmap.add_attr(attrpair.first,
				marsh::ObjptrT(opt::parse_attr(attrpair.second, *ginfo_)));
		}
		return std::make_shared<FunctorTarget>(opname, args, std::move(attrmap));
	}

	const opt::GraphInfo* ginfo_;
};

}

#endif // HONE_TARGET_HPP
