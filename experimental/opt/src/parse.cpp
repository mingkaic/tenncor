#include "experimental/opt/parse/def.h"

#include "experimental/opt/parse.hpp"

#ifdef OPT_PARSE_HPP

namespace opt
{

static void vectorize (std::vector<double>& arr, ::NumList* list)
{
	arr = {};
	for (auto it = list; NULL != it; it = it->next_)
	{
		arr.push_back(it->val_);
	}
}

static rule::WriterptrT make_writer (::Subgraph* sg,
	const std::unordered_map<std::string,size_t>& symbols)
{
	if (NULL == sg)
	{
		logs::fatal("cannot make writer with null subgraph");
	}
	rule::WriterptrT out;
	switch (sg->type_)
	{
		case SCALAR:
			out = std::make_shared<rule::Scalar>(sg->val_.scalar_);
			break;
		case ANY:
		{
			std::string symbol(sg->val_.any_);
			auto it = symbols.find(symbol);
			if (symbols.end() == it)
			{
				logs::fatalf("undeclared symbol '%s'", symbol.c_str());
			}
			out = std::make_shared<rule::Any>(it->second);
		}
			break;
		case BRANCH:
		{
			::Branch* branch = sg->val_.branch_;
			if (NULL == branch)
			{
				logs::fatal("subgraph ended at NULL branch");
			}
			rule::WriterArgsT args;
			for (auto it = branch->args_; NULL != it; it = it->next_)
			{
				::Arg* arg = it->val_;
				rule::WriterptrT warg = make_writer(arg->subgraph_, symbols);
				std::string shaper;
				std::string coorder;
				{
					std::vector<double> slist;
					std::vector<double> clist;
					vectorize(slist, arg->shaper_);
					vectorize(clist, arg->coorder_);
					if (slist.size() > 0)
					{
						shaper = fmts::join(",", slist.begin(), slist.end());
					}
					if (clist.size() > 0)
					{
						coorder = fmts::join(",", clist.begin(), clist.end());
					}
				}
				args.push_back(rule::WriterArg(warg, shaper, coorder));
			}
			std::string label(branch->label_);
			if (branch->is_group_)
			{
				size_t vid = std::string::npos;
				std::string variadic(branch->variadic_);
				auto vit = symbols.find(variadic);
				if (variadic.size() > 0 && symbols.end() != vit)
				{
					vid = vit->second;
				}
				out = std::make_shared<rule::Group>(label, args, vid);
			}
			else
			{
				out = std::make_shared<rule::Func>(label, args);
			}
		}
			break;
		default:
			logs::fatalf("unknown subgraph node type %d", sg->type_);
	}
	return out;
}

static rule::BuilderptrT make_builder (::Subgraph* sg,
	const std::unordered_map<std::string,size_t>& symbols)
{
	if (NULL == sg)
	{
		logs::fatal("cannot make builder with null subgraph");
	}
	rule::BuilderptrT out;
	return nullptr;
}

rule::ConversionsT parse (std::string filename)
{
	rule::ConversionsT conversions;
	::StmtList* stmts = NULL;
	int status = parse_rule(&stmts, filename.c_str());
	if (status != 0)
	{
		logs::errorf("failed to parse file %s: got %d status",
			filename.c_str(), status);
		return conversions;
	}
	std::unordered_map<std::string,size_t> symbols;

	for (auto it = stmts; it != NULL; it = it->next_)
	{
		switch (stmts->type_)
		{
			case SYMBOL_DEF:
			{
				char* symbol = (char*) stmts->val_;
				symbols.emplace(std::string(symbol), symbols.size());
			}
				break;
			case CONVERSION:
			{
				::Conversion* conv = (::Conversion*) stmts->val_;
				rule::WriterptrT writer = make_writer(conv->source_, symbols);
				rule::BuilderptrT builder = make_builder(conv->dest_, symbols);
				conversions.push_back(rule::Conversion(writer, builder));
			}
				break;
			default:
				logs::errorf("unknown statement of type %d", stmts->val_);
				return conversions;
		}
	}
	stmts_recursive_free(stmts);

	return conversions;
}

}

#endif // OPT_RULE_PARSE_HPP
