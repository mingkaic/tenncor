#include "ead/parse.hpp"

#ifdef EAD_PARSE_HPP

namespace ead
{

static void vectorize (std::vector<double>& arr, ::NumList* list)
{
	arr = {};
	for (auto it = list; NULL != it; it = it->next_)
	{
		arr.push_back(it->val_);
	}
}

ade::CoordptrT shaperize (::NumList* list)
{
	ade::CoordptrT out = nullptr;
	std::vector<double> slist;
	vectorize(slist, list);
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

CoordptrT coorderize (::NumList* list)
{
	CoordptrT out = nullptr;
	std::vector<double> clist;
	vectorize(clist, list);
	if (clist.size() > 0)
	{
		ade::CoordT carr;
		std::copy(clist.begin(), clist.end(), carr.begin());
		out = std::make_shared<CoordMap>(carr, false); // todo: figure out bijectivity
	}
	return out;
}

opt::rule::WriterptrT make_writer (
	::Subgraph* sg, const RuleContext& ctx)
{
	if (NULL == sg)
	{
		logs::fatal("cannot make writer with null subgraph");
	}
	opt::rule::WriterptrT out;
	switch (sg->type_)
	{
		case SCALAR:
			out = std::make_shared<opt::rule::ScalarWriter>(sg->val_.scalar_);
			break;
		case ANY:
		{
			std::string symbol(sg->val_.any_);
			auto it = ctx.symbols_.find(symbol);
			if (ctx.symbols_.end() == it)
			{
				logs::fatalf("undeclared symbol '%s'", symbol.c_str());
			}
			out = std::make_shared<opt::rule::AnyWriter>(it->second);
		}
			break;
		case BRANCH:
		{
			::Branch* branch = sg->val_.branch_;
			if (NULL == branch)
			{
				logs::fatal("subgraph ended at NULL branch");
			}
			opt::rule::WriterArgsT args;
			for (auto it = branch->args_; NULL != it; it = it->next_)
			{
				::Arg* arg = it->val_;
				opt::rule::WriterptrT warg = make_writer(arg->subgraph_, ctx);
				ade::CoordptrT shaper = shaperize(arg->shaper_);
				ade::CoordptrT coorder = coorderize(arg->coorder_);
				args.push_back(opt::rule::WriterArg(warg, shaper, coorder));
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
				size_t group_id = ctx.group_refs_.at(label);
				std::string group_tag = ctx.group_tags_.at(label);
				out = std::make_shared<opt::rule::GroupWriter>(
					group_id, group_tag, args, vid);
			}
			else
			{
				out = std::make_shared<opt::rule::FuncWriter>(label, args);
			}
		}
			break;
		default:
			logs::fatalf("unknown subgraph node type %d", sg->type_);
	}
	return out;
}

}

#endif
