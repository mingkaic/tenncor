#include "experimental/match/matcher.hpp"

namespace opt
{

namespace match
{

static boost::uuids::random_generator uuid_gen;

static const std::string group_prefix = "group:";

// commutative label according to .rules configuration
static const std::string commutative_prop = "commutative";

struct RuleContext final
{
	// maps declared symbol to any id
	std::unordered_map<std::string,size_t> symbols_;
	// maps declared group reference to group id
	std::unordered_map<std::string,size_t> group_refs_;
	// maps declared group reference to group tag
	std::unordered_map<std::string,std::string> group_tags_;
	// maps functor/group label to properties set
	std::unordered_map<std::string,std::unordered_set<std::string> properties_;
};

bool is_commutative (const RuleContext& ctx, std::string label)
{
	auto it = ctx.properties_.find(label);
	if (ctx.properties_.end() == it)
	{
		return false;
	}
	return util::has(it->second, commutative_prop);
}

ade::CoordptrT shaperize (::NumList* list)
{
	ade::CoordptrT out = nullptr;
	if (nullptr == list)
	{
		return out;
	}
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
	if (nullptr == list)
	{
		return out;
	}
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

// return label + whether type is ANY
std::string build_intermediate (
	VoterPool& voters, const ::Subgraph* sg, const RuleContext& ctx)
{
	std::string out;
    switch (sg->type_)
	{
		case SCALAR:
		{
			std::string scalar_id = fmts::to_string(sg->val_.scalar_);
			voters.immutables_.emplace(scalar_id);
			out = scalar_id;
		}
			break;
		case ANY:
		{
			std::string symbol(sg->val_.any_);
			auto it = ctx.symbols_.find(symbol);
			if (ctx.symbols_.end() == it)
			{
				logs::fatalf("undeclared symbol '%s'", symbol.c_str());
			}
			out = symbol;
		}
			break;
		case BRANCH:
		{
			::Branch* branch = sg->val_.branch_;
			if (nullptr == branch)
			{
				logs::fatal("subgraph ended at null branch");
			}
			std::string label(branch->label_);
			if (branch->is_group_)
			{
				label = group_prefix + label;
			}

			VoterArgsT args;
			for (auto it = branch->args_->head_; nullptr != it; it = it->next_)
			{
				::Arg* arg = (::Arg*) it->val_;
				std::string arg_label = build_intermediate(
					voters, arg->subgraph_, ctx);
				ade::CoordptrT shaper = shaperize(arg->shaper_);
				ade::CoordptrT coorder = coorderize(arg->coorder_);
				args.push_back(VoterArg{
					arg_label,
					shaper,
					coorder,
					arg->subgraph_->type_,
				});
			}
			if (util::has(voters.branches_, label))
			{
				if (is_commutative(ctx, label))
				{
					voters.branches_.emplace(label,
						std::make_unique<CommVoter>(label));
				}
				else
				{
					voters.branches_.emplace(label,
						std::make_unique<OrdrVoter>(label));
				}
			}
			std::string interm_id = boost::uuids::to_string(uuid_gen());
			voters.branches_[label]->emplace(args, Symbol{
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

void build_conversion (VoterPool& voters,
    const ::Conversion* conv, const RuleContext& ctx)
{
	if (NULL == conv)
	{
		logs::fatal("cannot make matcher with null conversion");
	}
	std::string conv_ref = boost::uuids::to_string(uuid_gen_());
	voters.converts_.emplace(conv_ref, conv->dest_);

	const ::Subgraph* sg = conv->source_;
	if (BRANCH == sg->type_)
	{
		::Branch* branch = sg->val_.branch_;
		if (nullptr == branch)
		{
			logs::fatal("conversion ended at nullptr branch");
		}
		std::string label(branch->label_);
		if (branch->is_group_)
		{
			label = group_prefix + label;
		}

		VoterArgsT args;
		for (auto it = branch->args_->head_; nullptr != it; it = it->next_)
		{
			::Arg* arg = (::Arg*) it->val_;
			std::string arg_label = build_intermediate(
				voters, arg->subgraph_, ctx);
			ade::CoordptrT shaper = shaperize(arg->shaper_);
			CoordptrT coorder = coorderize(arg->coorder_);
			args.push_back(MatchArg{
				arg_label,
				shaper,
				coorder,
				arg->subgraph_->type_,
			});
		}
		if (util::has(voters.branches_, label))
		{
			if (is_commutative(ctx, label))
			{
				voters.branches_.emplace(label,
					std::make_unique<CommVoter>(label));
			}
			else
			{
				voters.branches_.emplace(label,
					std::make_unique<OrdrVoter>(label));
			}
		}
		voters.branches_[label]->emplace(args, Symbol{
			CONVRT,
			conv_ref,
		});
	}
	else
	{
		logs::warn("ambiguous conversion...");
	}
}

VoterPool parse (std::string filename)
{
    VoterPool voters;
	::PtrList* stmts = nullptr;
    FILE* file = std::fopen(filename.c_str(), "r");
	if (nullptr == file)
	{
		logs::errorf("failed to open file %s", filename.c_str());
		return voters;
	}
	int status = ::parse_file(&stmts, file);
	if (status != 0)
	{
		logs::errorf("failed to parse file %s: got %d status",
			filename.c_str(), status);
		return voters;
	}

	if (nullptr == stmts)
	{
		logs::fatal("rule parser produced null stmts");
	}

	RuleContext ctx;
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
				ctx.symbols_.emplace(symbol, ctx.symbols_.size());
			}
				break;
			case GROUP_DEF:
			{
				::Group* group = (::Group*) stmt->val_;
				std::string ref = std::string(group->ref_);
				if (util::has(ctx.group_refs_, ref))
				{
					logs::fatalf("redeclaration of group %s", ref.c_str());
				}
				ctx.group_refs_.emplace(ref, ctx.group_refs_.size());
				ctx.group_tags_.emplace(ref, std::string(group->tag_));
			}
				break;
			case PROPERTY_DEF:
			{
				::Property* property = (::Property*) stmt->val_;
				std::string label = std::string(property->label_);
				std::string property = std::string(property->property_);
				if (property->is_group_)
				{
					label = group_prefix + label;
				}
				auto it = ctx.properties_.find(label);
				if (ctx.properties_.end() == it)
				{
					ctx.properties_.emplace(label, {property});
				}
				else if (util::has(it->second, property))
				{
					logs::warnf("reassignment of property %s to %s",
						property.c_str(), label.c_str());
				}
				else
				{
					it->second.emplace(property);
				}
			}
				break;
			case CONVERSION:
			{
				::Conversion* conv = (::Conversion*) stmt->val_;
				build_conversion(voters, conv, ctx);
			}
				break;
			default:
				logs::errorf("unknown statement of type %d", stmt->type_);
		}
	}
	::statements_free(stmts);

    return voters;
}

}

}
