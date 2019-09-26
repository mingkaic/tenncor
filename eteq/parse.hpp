#include "opt/parse.hpp"

#include "eteq/eteq.hpp"

#ifndef ETEQ_PARSE_HPP
#define ETEQ_PARSE_HPP

namespace eteq
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
		teq::CoordT carr;
		std::copy(clist.begin(), clist.end(), carr.begin());
		out = std::make_shared<CoordMap>(carr, false); // todo: figure out bijectivity
	}
	return out;
}

template <typename T>
struct ScalarConvr final : public opt::iConverter
{
	ScalarConvr (double scalar) : scalar_(scalar) {}

	teq::TensptrT build (const opt::ContexT& ctx,
		teq::Shape outshape) const override
	{
		return make_constant_scalar((T) scalar_, outshape)->get_tensor();
	}

	std::string to_string (void) const override
	{
		return fmts::to_string(scalar_);
	}

	double scalar_;
};

template <typename T>
struct AnyConvr final : public opt::iConverter
{
	AnyConvr (std::string any_id) : any_id_(any_id) {}

	teq::TensptrT build (const opt::ContexT& ctx,
		teq::Shape outshape) const override
	{
		const opt::CtxValT& val = estd::must_getf(ctx, any_id_,
			"cannot find any id `%s` in conversion", any_id_.c_str());
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
		teq::CoordptrT shaper, CoordptrT coorder) :
		arg_(arg), shaper_(shaper), coorder_(coorder)
	{
		if (nullptr == arg)
		{
			logs::fatal("created a rule argument with null argument");
		}
	}

	opt::ConvptrT arg_;

	teq::CoordptrT shaper_;

	CoordptrT coorder_;
};

using BuilderArgsT = std::vector<BuilderArg>;

template <typename T>
struct FuncConvr final : public opt::iConverter
{
	FuncConvr (std::string op, BuilderArgsT args) :
		opcode_({op, egen::get_op(op)}), args_(args) {}

	teq::TensptrT build (const opt::ContexT& ctx,
		teq::Shape outshape) const override
	{
		ArgsT<T> args;
		for (auto& arg : args_)
		{
			teq::Shape childshape = outshape;
			if (teq::is_identity(arg.shaper_.get()))
			{
				childshape = teq::apply_shaper(arg.shaper_, childshape);
			}
			auto tens = arg.arg_->build(ctx, childshape);
			args.push_back(FuncArg<T>(
				TO_NODE(tens),
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

	teq::Opcode opcode_;

	BuilderArgsT args_;
};

template <typename T>
struct GroupConvr final : public opt::iConverter
{
	GroupConvr (std::string group, BuilderArgsT args, std::string variadic) :
		group_(group), args_(args), variadic_(variadic)
	{
		assert(group_ == "sum" || group_ == "prod"); // todo: generalize this for ordered-groups
	}

	teq::TensptrT build (const opt::ContexT& ctx,
		teq::Shape outshape) const override
	{
		teq::TensT args;
		for (auto& arg : args_)
		{
			teq::Shape childshape = outshape;
			if (teq::is_identity(arg.shaper_.get()))
			{
				childshape = teq::apply_shaper(arg.shaper_, childshape);
			}
			args.push_back(arg.arg_->build(ctx, childshape));
		}

		if (variadic_.size() > 0)
		{
			opt::CtxValT varargs = estd::try_get(
				ctx, variadic_, opt::CtxValT());
			args.insert(args.end(), varargs.begin(), varargs.end());
		}
		eteq::NodesT<T> outs;
		outs.reserve(args.size());
		std::transform(args.begin(), args.end(),
			std::back_inserter(outs),
			[](teq::TensptrT tens)
			{
				return TO_NODE(tens);
			});
		if (group_ == "sum")
		{
			return tenncor::sum(outs)->get_tensor();
		}
		return tenncor::prod(outs)->get_tensor();
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
		return fmts::sprintf("group:%s(%s)", group_.c_str(),
			fmts::join(",", args.begin(), args.end()).c_str());
	}

	std::string group_;

	BuilderArgsT args_;

	std::string variadic_;
};

template <typename T>
struct ConverterBuilder final : public opt::iConverterBuilder
{
	opt::CstConvertF build_cconv (void) const override
	{
		return [](teq::iTensor* tens)
		{
			teq::TensptrT out = nullptr;
			if (auto f = dynamic_cast<teq::iOperableFunc*>(tens))
			{
				f->update();
				T* data = (T*) f->data();
				out = make_constant(data, tens->shape())->get_tensor();
			}
			return out;
		};
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
				if (false == estd::has(ctx.symbols_, symbol))
				{
					logs::fatalf("undeclared symbol `%s`", symbol.c_str());
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
					teq::CoordptrT shaper = this->shaperize(arg->shaper_);
					CoordptrT coorder = eteq::coorderize(arg->coorder_);
					args.push_back(BuilderArg(warg, shaper, coorder));
				}
				std::string label(branch->label_);
				if (branch->is_group_)
				{
					std::string variadic(branch->variadic_);
					if (variadic.size() > 0 && false == estd::has(ctx.symbols_, variadic))
					{
						logs::warnf("unknown variadic %s", variadic.c_str());
						variadic = "";
					}
					out = std::make_shared<GroupConvr<T>>(
						label, args, variadic);
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

	teq::CoordptrT shaperize (::NumList* list) const override
	{
		teq::CoordptrT out = nullptr;
		if (nullptr == list)
		{
			return out;
		}
		std::vector<double> slist = vectorize(list);
		if (slist.size() > 0)
		{
			out = std::make_shared<teq::CoordMap>(
			[&slist](teq::MatrixT m)
			{
				for (size_t i = 0; i < teq::mat_dim; ++i)
				{
					for (size_t j = 0; j < teq::mat_dim; ++j)
					{
						size_t index = i * teq::mat_dim + j;
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

	teq::CoordptrT coorderize (::NumList* list) const override
	{
		return eteq::coorderize(list);
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

#endif // ETEQ_PARSE_HPP