#include "opt/parse.hpp"

#include "ead/ead.hpp"

#ifndef EAD_PARSE_HPP
#define EAD_PARSE_HPP

namespace ead
{

static std::vector<double> vectorize (::NumList* list)
{
	std::vector<double> arr;
	for (auto it = list->head_; nullptr != it; it = it->next_)
	{
		arr.push_back(it->val_);
	}
	return arr;
}

static CoordptrT coorderize (::NumList* list)
{
	CoordptrT out = nullptr;
	if (nullptr == list)
	{
		return out;
	}
	std::vector<double> clist = vectorize(list);
	if (clist.size() > 0)
	{
		ade::CoordT carr;
		std::copy(clist.begin(), clist.end(), carr.begin());
		out = std::make_shared<CoordMap>(carr, false); // todo: figure out bijectivity
	}
	return out;
}

template <typename T>
struct ScalarConvr final : public opt::iConverter
{
	ScalarConvr (double scalar) : scalar_(scalar) {}

	ade::TensptrT build (const opt::ContexT& ctx,
		ade::Shape outshape) const override
	{
		return ead::make_constant_scalar((T) scalar_,
			outshape)->get_tensor();
	}

	std::string to_string (void) const override
	{
		return fmts::to_string(scalar_);
	}

	double scalar_;
};

template <typename T>
struct ConstConvr final : public opt::iConverter
{
	ade::TensptrT build (const opt::ContexT& ctx,
		ade::Shape outshape) const override
	{
		if (ctx.size() != 1)
		{
			logs::fatal("cannot build constant from multiple context");
		}
		ade::TensptrT tens = *(ctx.begin()->second.begin());
		if (auto f = static_cast<ade::iOperableFunc*>(tens.get()))
		{
			f->update();
			T* data = (T*) f->data();
			tens = ead::make_constant(data, outshape)->get_tensor();
		}
		return tens;
	}

	std::string to_string (void) const override
	{
		return "ConstConvr";
	}
};

template <typename T>
struct AnyConvr final : public opt::iConverter
{
	AnyConvr (std::string any_id) : any_id_(any_id) {}

	ade::TensptrT build (const opt::ContexT& ctx,
		ade::Shape outshape) const override
	{
		auto it = ctx.find(any_id_);
		if (ctx.end() == it)
		{
			logs::fatalf("cannot find any id %s in conversion",
				any_id_.c_str());
		}
		const opt::CtxValT& val = it->second;
		if (val.size() != 1)
		{
			logs::fatal("context value is not any");
		}
		return *(val.begin());
	}

	std::string to_string (void) const override
	{
		return any_id_;
	}

	std::string any_id_;
};

struct BuilderArg final
{
	BuilderArg (opt::ConvptrT arg,
		ade::CoordptrT shaper, CoordptrT coorder) :
		arg_(arg), shaper_(shaper), coorder_(coorder)
	{
		if (nullptr == arg)
		{
			logs::fatal("created a rule argument with null argument");
		}
	}

	opt::ConvptrT arg_;

	ade::CoordptrT shaper_;

	CoordptrT coorder_;
};

using BuilderArgsT = std::vector<BuilderArg>;

template <typename T>
struct FuncConvr final : public opt::iConverter
{
	FuncConvr (std::string op, BuilderArgsT args) :
		opcode_({op, age::get_op(op)}), args_(args) {}

	ade::TensptrT build (const opt::ContexT& ctx,
		ade::Shape outshape) const override
	{
		ArgsT<T> args;
		for (auto& arg : args_)
		{
			ade::Shape childshape = outshape;
			if (ade::is_identity(arg.shaper_.get()))
			{
				childshape = ade::apply_shaper(arg.shaper_, childshape);
			}
			auto tens = arg.arg_->build(ctx, childshape);
			args.push_back(FuncArg<T>(
				NodeConverters<T>::to_node(tens),
				arg.shaper_, arg.coorder_));
		}
		return make_functor(opcode_, args)->get_tensor();
	}

	std::string to_string (void) const override
	{
		std::vector<std::string> args;
		args.reserve(args_.size());
		std::transform(args_.begin(), args_.end(),
			std::back_inserter(args),
			[](const BuilderArg& arg)
			{
				return arg.arg_->to_string();
			});
		return opcode_.name_ + fmts::sprintf("(%s)", fmts::join(",",
			args.begin(), args.end()).c_str());
	}

	ade::Opcode opcode_;

	BuilderArgsT args_;
};

template <typename T>
struct GroupConvr final : public opt::iConverter
{
	GroupConvr (std::string group_id, std::string group,
		BuilderArgsT args, std::string variadic) :
		group_id_(group_id), group_(group), args_(args), variadic_(variadic)
	{
		assert(group_ == "sum" || group_ == "prod"); // todo: generalize this for ordered-groups
	}

	ade::TensptrT build (const opt::ContexT& ctx,
		ade::Shape outshape) const override
	{
		auto git = ctx.find(group_id_);
		if (ctx.end() == git)
		{
			logs::fatalf("cannot find group id %s in conversion",
				group_id_.c_str());
		}
		ade::TensptrT group_head = *(git->second.begin());

		ade::TensT args;
		for (auto& arg : args_)
		{
			ade::Shape childshape = outshape;
			if (ade::is_identity(arg.shaper_.get()))
			{
				childshape = ade::apply_shaper(arg.shaper_, childshape);
			}
			args.push_back(arg.arg_->build(ctx, childshape));
		}

		if (variadic_.size() > 0)
		{
			auto it = ctx.find(variadic_);
			if (ctx.end() != it)
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
		if (group_ == "sum")
		{
			return age::sum(outs)->get_tensor();
		}
		return age::prod(outs)->get_tensor();
	}

	std::string to_string (void) const override
	{
		std::vector<std::string> args;
		args.reserve(args_.size());
		std::transform(args_.begin(), args_.end(),
			std::back_inserter(args),
			[](const BuilderArg& arg)
			{
				return arg.arg_->to_string();
			});
		if (variadic_.size() > 0)
		{
			args.push_back(".." + variadic_);
		}
		return fmts::sprintf("group:%s(%s)", group_id_.c_str(),
			fmts::join(",", args.begin(), args.end()).c_str());
	}

	std::string group_id_;

	std::string group_;

	BuilderArgsT args_;

	std::string variadic_;
};

template <typename T>
struct ConverterBuilder final : public opt::iConverterBuilder
{
	opt::ConvptrT build_cconv (void) const override
	{
		return std::make_shared<ConstConvr<T>>();
	}

	opt::ConvptrT build (const ::Subgraph* sg,
		const opt::RulesContext& ctx) const override
	{
		if (NULL == sg)
		{
			logs::fatal("cannot make builder with null subgraph");
		}
		opt::ConvptrT out;
		switch (sg->type_)
		{
			case SCALAR:
				out = std::make_shared<ScalarConvr<T>>(sg->val_.scalar_);
				break;
			case ANY:
			{
				std::string symbol(sg->val_.any_);
				if (false == util::has(ctx.symbols_, symbol))
				{
					logs::fatalf("undeclared symbol '%s'", symbol.c_str());
				}
				out = std::make_shared<AnyConvr<T>>(symbol);
			}
				break;
			case BRANCH:
			{
				::Branch* branch = sg->val_.branch_;
				if (nullptr == branch)
				{
					logs::fatal("subgraph ended at nullptr branch");
				}
				BuilderArgsT args;
				for (auto it = branch->args_->head_; nullptr != it; it = it->next_)
				{
					::Arg* arg = (::Arg*) it->val_;
					opt::ConvptrT warg = build(arg->subgraph_, ctx);
					ade::CoordptrT shaper = this->shaperize(arg->shaper_);
					CoordptrT coorder = ead::coorderize(arg->coorder_);
					args.push_back(BuilderArg(warg, shaper, coorder));
				}
				std::string label(branch->label_);
				if (branch->is_group_)
				{
					std::string variadic(branch->variadic_);
					if (variadic.size() > 0 && false == util::has(ctx.symbols_, variadic))
					{
						logs::warnf("unknown variadic %s", variadic.c_str());
						variadic = "";
					}
					auto group_it = ctx.group_tags_.find(label);
					if (ctx.group_tags_.end() == group_it)
					{
						logs::fatalf("cannot find ref %s", label.c_str());
					}
					out = std::make_shared<GroupConvr<T>>(
						label, group_it->second, args, variadic);
				}
				else
				{
					out = std::make_shared<FuncConvr<T>>(label, args);
				}
			}
				break;
			default:
				logs::fatalf("unknown subgraph node type %d", sg->type_);
		}
		return out;
	}

	ade::CoordptrT shaperize (::NumList* list) const override
	{
		ade::CoordptrT out = nullptr;
		if (nullptr == list)
		{
			return out;
		}
		std::vector<double> slist = vectorize(list);
		if (slist.size() > 0)
		{
			out = std::make_shared<ade::CoordMap>(
			[&slist](ade::MatrixT m)
			{
				for (size_t i = 0; i < ade::mat_dim; ++i)
				{
					for (size_t j = 0; j < ade::mat_dim; ++j)
					{
						size_t index = i * ade::mat_dim + j;
						if (index < slist.size())
						{
							m[i][j] = slist[index];
						}
					}
				}
			});
		}
		return out;
	}

	ade::CoordptrT coorderize (::NumList* list) const override
	{
		return ead::coorderize(list);
	}
};

template <typename T>
opt::OptCtx parse (std::string content)
{
	static ConverterBuilder<T> builder;
	return opt::parse(content, builder);
}

template <typename T>
opt::OptCtx parse_file (std::string filename)
{
	static ConverterBuilder<T> builder;
	return opt::parse_file(filename, builder);
}

}

#endif // EAD_PARSE_HPP
