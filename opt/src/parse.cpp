#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "opt/voter.hpp"
#include "opt/parse.hpp"

#ifdef OPT_PARSE_HPP

namespace opt
{

static boost::uuids::random_generator uuid_gen;

// commutative label according to .rules configuration
static const std::string commutative_prop = "commutative";

static bool is_commutative (const RulesContext& ctx, std::string label)
{
	auto it = ctx.properties_.find(label);
	if (ctx.properties_.end() == it)
	{
		return false;
	}
	return util::has(it->second, commutative_prop);
}

static std::string build_intermediate (VoterPool& voters, const ::Subgraph* sg,
	const RulesContext& ctx, const iConverterBuilder& builder);

static std::string build_args (VoterPool& voters,
	VoterArgsT& args, const ::Branch* branch,
	const RulesContext& ctx, const iConverterBuilder& builder)
{
	if (nullptr == branch)
	{
		logs::fatal("subgraph ended at null branch");
	}
	std::string label(branch->label_);
	if (branch->is_group_)
	{
		label = group_prefix + label;
	}

	for (auto it = branch->args_->head_; nullptr != it; it = it->next_)
	{
		::Arg* arg = (::Arg*) it->val_;
		std::string arg_label = build_intermediate(
			voters, arg->subgraph_, ctx, builder);
		args.push_back(VoterArg{
			arg_label,
			builder.shaperize(arg->shaper_),
			builder.coorderize(arg->coorder_),
			arg->subgraph_->type_,
		});
	}
	if (false == util::has(voters.branches_, label))
	{
		if (is_commutative(ctx, label))
		{
			std::string variadic(branch->variadic_);
			if (variadic.size() > 0 && branch->is_group_ &&
				util::has(ctx.symbols_, variadic))
			{
				voters.branches_.emplace(label,
					std::make_shared<VariadicVoter>(label, variadic));
			}
			else
			{
				voters.branches_.emplace(label,
					std::make_shared<CommVoter>(label));
			}
		}
		else
		{
			voters.branches_.emplace(label,
				std::make_shared<OrdrVoter>(label));
		}
	}
	return label;
}

// return label + whether type is ANY
static std::string build_intermediate (VoterPool& voters, const ::Subgraph* sg,
	const RulesContext& ctx, const iConverterBuilder& builder)
{
	std::string out;
	switch (sg->type_)
	{
		case ::SUBGRAPH_TYPE::SCALAR:
		{
			std::string scalar_id = fmts::to_string(sg->val_.scalar_);
			voters.immutables_.emplace(scalar_id);
			out = scalar_id;
		}
			break;
		case ::SUBGRAPH_TYPE::ANY:
		{
			std::string symbol(sg->val_.any_);
			if (false == util::has(ctx.symbols_, symbol))
			{
				logs::fatalf("undeclared symbol '%s'", symbol.c_str());
			}
			out = symbol;
		}
			break;
		case ::SUBGRAPH_TYPE::BRANCH:
		{
			VoterArgsT args;
			std::string label = build_args(
				voters, args, sg->val_.branch_, ctx, builder);

			std::string interm_id = boost::uuids::to_string(uuid_gen());
			voters.branches_[label]->emplace(args,
				Symbol{
					INTERM,
					interm_id,
				});
			out = interm_id;
		}
			break;
		default:
			logs::fatalf("unknown subgraph node type %d", sg->type_);
	}
	return out;
}

static void build_conversion (OptCtx& opts, const ::Conversion* conv,
	const RulesContext& ctx, const iConverterBuilder& builder)
{
	if (NULL == conv)
	{
		logs::fatal("cannot make matcher with null conversion");
	}
	std::string conv_ref = boost::uuids::to_string(uuid_gen());
	opts.converts_.emplace(conv_ref, builder.build(conv->dest_, ctx));

	const ::Subgraph* sg = conv->source_;
	if (BRANCH == sg->type_)
	{
		VoterArgsT args;
		std::string label = build_args(
			opts.voters_, args, sg->val_.branch_, ctx, builder);

		opts.voters_.branches_[label]->emplace(args,
			Symbol{
				CONVRT,
				conv_ref,
			});
	}
	else
	{
		logs::warn("ambiguous conversion...");
	}
}

OptCtx process_stmts (::PtrList* stmts, const iConverterBuilder& builder)
{
	OptCtx opts;
	if (nullptr == stmts)
	{
		logs::fatal("rule parser produced null stmts");
	}
	RulesContext ctx;
	for (auto it = stmts->head_; it != NULL; it = it->next_)
	{
		::Statement* stmt = (::Statement*) it->val_;
		switch (stmt->type_)
		{
			case SYMBOL_DEF:
			{
				std::string symbol = std::string((char*) stmt->val_);
				if (util::has(ctx.symbols_, symbol))
				{
					logs::fatalf("redeclaration of symbol %s", symbol.c_str());
				}
				ctx.symbols_.emplace(symbol);
			}
				break;
			case GROUP_DEF:
			{
				::Group* group = (::Group*) stmt->val_;
				std::string ref = std::string(group->ref_);
				if (util::has(ctx.group_tags_, ref))
				{
					logs::fatalf("redeclaration of group %s", ref.c_str());
				}
				ctx.group_tags_.emplace(ref, std::string(group->tag_));
			}
				break;
			case PROPERTY_DEF:
			{
				::Property* property = (::Property*) stmt->val_;
				std::string label = std::string(property->label_);
				std::string property_tag = std::string(property->property_);
				if (property->is_group_)
				{
					label = group_prefix + label;
				}
				auto it = ctx.properties_.find(label);
				if (ctx.properties_.end() == it)
				{
					ctx.properties_.emplace(label,
						std::unordered_set<std::string>{property_tag});
				}
				else if (util::has(it->second, property_tag))
				{
					logs::warnf("reassignment of property %s to %s",
						property_tag.c_str(), label.c_str());
				}
				else
				{
					it->second.emplace(property_tag);
				}
			}
				break;
			case CONVERSION:
			{
				::Conversion* conv = (::Conversion*) stmt->val_;
				build_conversion(opts, conv, ctx, builder);
			}
				break;
			default:
				logs::errorf("unknown statement of type %d", stmt->type_);
		}
	}
	::statements_free(stmts);
	return opts;
}

OptCtx parse (std::string content, const iConverterBuilder& builder)
{
	::PtrList* stmts = nullptr;
	int status = ::parse_str(&stmts, content.c_str());
	if (status != 0)
	{
		logs::errorf("failed to parse content %s: got %d status",
			content.c_str(), status);
		return OptCtx();
	}
	return process_stmts(stmts, builder);
}

OptCtx parse_file (std::string filename, const iConverterBuilder& builder)
{
	::PtrList* stmts = nullptr;
	FILE* file = std::fopen(filename.c_str(), "r");
	if (nullptr == file)
	{
		logs::errorf("failed to open file %s", filename.c_str());
		return OptCtx();
	}
	int status = ::parse_file(&stmts, file);
	if (status != 0)
	{
		logs::errorf("failed to parse file %s: got %d status",
			filename.c_str(), status);
		return OptCtx();
	}
	return process_stmts(stmts, builder);
}

}

#endif
