#include "experimental/opt/rule/convert.hpp"

extern "C" {
#include "experimental/opt/parse/def.h"
}

#include "ead/ead.hpp"

#ifndef EAD_PARSE_HPP
#define EAD_PARSE_HPP

namespace ead
{

template <typename T>
struct ScalarBuilder final : public opt::rule::iBuilder
{
	ScalarBuilder (double scalar) : scalar_(scalar) {}

	ade::TensptrT build (opt::rule::Report& report,
		ade::Shape outshape) override
	{
		return ead::make_constant_scalar((T) scalar_,
			outshape)->get_tensor();
	}

	double scalar_;
};

template <typename T>
struct AnyBuilder final : public opt::rule::iBuilder
{
	AnyBuilder (size_t any_id) : any_id_(any_id) {}

	ade::TensptrT build (opt::rule::Report& report,
		ade::Shape outshape) override
	{
		auto it = report.bases_.find(any_id_);
		if (report.bases_.end() == it)
		{
			logs::fatalf("cannot find any id %d in conversion", any_id_);
		}
		return it->second;
	}

	size_t any_id_;
};

struct BuilderArg final
{
	BuilderArg (opt::rule::BuilderptrT arg,
		ade::CoordptrT shaper, CoordptrT coorder) :
		arg_(arg), shaper_(shaper), coorder_(coorder)
	{
		if (nullptr == arg)
		{
			logs::fatal("created a rule argument with null argument");
		}
	}

	opt::rule::BuilderptrT arg_;

	ade::CoordptrT shaper_;

	CoordptrT coorder_;
};

using BuilderArgsT = std::vector<BuilderArg>;

template <typename T>
struct FuncBuilder final : public opt::rule::iBuilder
{
	FuncBuilder (std::string op, BuilderArgsT args) :
		opcode_({op, age::get_op(op)}), args_(args) {}

	ade::TensptrT build (opt::rule::Report& report,
		ade::Shape outshape) override
	{
		ArgsT<T> args;
		for (auto& arg : args_)
		{
			ade::Shape childshape = outshape;
			if (ade::is_identity(arg.shaper_.get()))
			{
				childshape = ade::apply_shaper(arg.shaper_, childshape);
			}
			auto tens = arg.arg_->build(report, childshape);
			args.push_back(FuncArg<T>(
				NodeConverters<T>::to_node(tens),
				arg.shaper_, arg.coorder_));
		}
		return make_functor(opcode_, args)->get_tensor();
	}

	ade::Opcode opcode_;

	BuilderArgsT args_;
};

template <typename T>
struct GroupBuilder final : public opt::rule::iBuilder
{
	GroupBuilder (size_t group_id, BuilderArgsT args, size_t variadic) :
		group_id_(group_id), args_(args), variadic_(variadic) {}

	ade::TensptrT build (opt::rule::Report& report,
		ade::Shape outshape) override
	{
		auto git = report.groups_.find(group_id_);
		if (report.groups_.end() == git)
		{
			logs::fatalf("cannot find group id %s in conversion", group_id_);
		}
		tag::SgraphptrT sg = git->second;
		assert(sg->group_ == "sum" || sg->group_ == "prod"); // todo: generalize this for ordered-groups

		ade::TensT args;
		for (auto& arg : args_)
		{
			ade::Shape childshape = outshape;
			if (ade::is_identity(arg.shaper_.get()))
			{
				childshape = ade::apply_shaper(arg.shaper_, childshape);
			}
			args.push_back(arg.arg_->build(report, childshape));
		}

		{
			auto it = report.variadics_.find(variadic_);
			if (report.variadics_.end() != it)
			{
				args.insert(args.end(),
					it->second.begin(), it->second.end());
			}
		}
		ead::NodesT<T> outs;
		outs.reserve(args.size());
		std::transform(args.begin(), args.end(),
			std::back_inserter(outs),
			[](ade::TensptrT tens)
			{
				return NodeConverters<T>::to_node(tens);
			});
		if (sg->group_ == "sum")
		{
			return age::sum(outs)->get_tensor();
		}
		return age::prod(outs)->get_tensor();
	}

	size_t group_id_;

	BuilderArgsT args_;

	size_t variadic_;
};

struct RuleContext final
{
	// maps declared symbol to any id
	std::unordered_map<std::string,size_t> symbols_;
	// maps declared group reference to group id
	std::unordered_map<std::string,size_t> group_refs_;
	// maps declared group reference to group tag
	std::unordered_map<std::string,std::string> group_tags_;
};

ade::CoordptrT shaperize (::NumList* list);

CoordptrT coorderize (::NumList* list);

opt::rule::WriterptrT make_writer (
	::Subgraph* sg, const RuleContext& ctx);

template <typename T>
opt::rule::BuilderptrT make_builder (
	::Subgraph* sg, const RuleContext& ctx)
{
	if (NULL == sg)
	{
		logs::fatal("cannot make builder with null subgraph");
	}
	opt::rule::BuilderptrT out;
	switch (sg->type_)
	{
		case SCALAR:
			out = std::make_shared<ScalarBuilder<T>>(sg->val_.scalar_);
			break;
		case ANY:
		{
			std::string symbol(sg->val_.any_);
			auto it = ctx.symbols_.find(symbol);
			if (ctx.symbols_.end() == it)
			{
				logs::fatalf("undeclared symbol '%s'", symbol.c_str());
			}
			out = std::make_shared<AnyBuilder<T>>(it->second);
		}
			break;
		case BRANCH:
		{
			::Branch* branch = sg->val_.branch_;
			if (NULL == branch)
			{
				logs::fatal("subgraph ended at NULL branch");
			}
			BuilderArgsT args;
			for (auto it = branch->args_; NULL != it; it = it->next_)
			{
				::Arg* arg = it->val_;
				opt::rule::BuilderptrT warg = make_builder<T>(arg->subgraph_, ctx);
				ade::CoordptrT shaper = shaperize(arg->shaper_);
				CoordptrT coorder = coorderize(arg->coorder_);
				args.push_back(BuilderArg(warg, shaper, coorder));
			}
			std::string label(branch->label_);
			if (branch->is_group_)
			{
				size_t vid = std::string::npos;
				std::string variadic(branch->variadic_);
				auto vit = ctx.symbols_.find(variadic);
				if (variadic.size() > 0 && ctx.symbols_.end() != vit)
				{
					vid = vit->second;
				}
				auto group_it = ctx.group_refs_.find(label);
				if (ctx.group_refs_.end() == group_it)
				{
					logs::fatalf("cannot find ref %s", label.c_str());
				}
				out = std::make_shared<GroupBuilder<T>>(
					group_it->second, args, vid);
			}
			else
			{
				out = std::make_shared<FuncBuilder<T>>(label, args);
			}
		}
			break;
		default:
			logs::fatalf("unknown subgraph node type %d", sg->type_);
	}
	return out;
}

template <typename T>
opt::rule::ConversionsT parse (std::string filename)
{
	opt::rule::ConversionsT conversions;
	::StmtList* stmts = NULL;
	int status = ::parse_rule(&stmts, filename.c_str());
	if (status != 0)
	{
		logs::errorf("failed to parse file %s: got %d status",
			filename.c_str(), status);
		return conversions;
	}

	RuleContext ctx;
	for (auto it = stmts; it != NULL; it = it->next_)
	{
		switch (stmts->type_)
		{
			case SYMBOL_DEF:
			{
				std::string symbol = std::string((char*) stmts->val_);
				if (util::has(ctx.symbols_, symbol))
				{
					logs::fatalf("redeclaration of symbol %s", symbol.c_str());
				}
				ctx.symbols_.emplace(symbol, ctx.symbols_.size());
			}
				break;
			case GROUP_DEF:
			{
				struct Group* group = (struct Group*) stmts->val_;
				std::string ref = std::string(group->ref_);
				if (util::has(ctx.group_refs_, ref))
				{
					logs::fatalf("redeclaration of group %s", ref.c_str());
				}
				ctx.group_refs_.emplace(ref, ctx.group_refs_.size());
				ctx.group_tags_.emplace(ref, std::string(group->tag_));
			}
				break;
			case CONVERSION:
			{
				::Conversion* conv = (::Conversion*) stmts->val_;
				opt::rule::WriterptrT writer = make_writer(conv->source_, ctx);
				opt::rule::BuilderptrT builder = make_builder<T>(conv->dest_, ctx);
				conversions.push_back(opt::rule::Conversion(writer, builder));
			}
				break;
			default:
				logs::errorf("unknown statement of type %d", stmts->val_);
				return conversions;
		}
	}
	::stmts_recursive_free(stmts);

	return conversions;
}

}

#endif // EAD_PARSE_HPP
