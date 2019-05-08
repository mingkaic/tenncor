#include "ead/opt/rule_src.hpp"
#include "ead/opt/ops_prune.hpp" // todo: move functionality here

#ifndef EAD_CONVERSION_HPP
#define EAD_CONVERSION_HPP

namespace ead
{

namespace opt
{

template <typename T>
using OwnerMapT = std::unordered_map<iReprNode<T>*,RepptrT<T>>;

template <typename T>
struct RuleTarget
{
	virtual ~RuleTarget (void) = default;

	/// Return representation node defined by this target-rule
	/// Can return nullptr if representation holds no value
	virtual RepptrT<T> make_rep (Representer<T>& repr,
		OwnerMapT<T>& owners, const RuleContext<T>& ctx,
		ade::Shape shape) const = 0;
};

template <typename T>
using RuleTargetptrT = std::shared_ptr<RuleTarget<T>>;

template <typename T>
struct TargetArg final
{
	RuleTargetptrT<T> arg_;

	ade::CoordptrT shaper_;

	ade::CoordptrT coorder_;
};

template <typename T>
using TargetArgsT = std::vector<TargetArg<T>>;

template <typename T>
struct VarTarget final : public RuleTarget<T>
{
	VarTarget (size_t id) : id_(id) {}

	RepptrT<T> make_rep (Representer<T>& repr,
		OwnerMapT<T>& owners, const RuleContext<T>& ctx,
		ade::Shape shape) const override
	{
		auto var_it = ctx.rule_vars_.find(id_);
		assert(ctx.rule_vars_.end() != var_it);
		auto owner_it = owners.find(var_it->second);
		assert(owners.end() != owner_it);
		return owner_it->second;
	}

	size_t id_;
};

template <typename T>
struct ScalarTarget final : public RuleTarget<T>
{
	ScalarTarget (T scalar) : scalar_(scalar) {}

	RepptrT<T> make_rep (Representer<T>& repr,
		OwnerMapT<T>& owners, const RuleContext<T>& ctx,
		ade::Shape shape) const override
	{
		std::vector<T> data(shape.n_elems(), scalar_);
		auto out = std::make_shared<ConstRep<T>>(data.data(), shape);
		owners.emplace(out.get(), out);
		return out;
	}

	T scalar_;
};

template <typename T>
struct FuncTarget final : public RuleTarget<T>
{
	FuncTarget (ade::Opcode opcode, TargetArgsT<T> args,
		std::vector<size_t> variadics) : opcode_(opcode),
		args_(args), variadics_(variadics) {}

	RepptrT<T> make_rep (Representer<T>& repr,
		OwnerMapT<T>& owners, const RuleContext<T>& ctx,
		ade::Shape shape) const override
	{
		RepArgsT<T> out_args;
		out_args.reserve(args_.size());
		for (const TargetArg<T>& arg : args_)
		{
			ade::Shape arg_shape = shape;
			auto& shaper = arg.shaper_;
			if (nullptr != shaper && ade::identity != shaper)
			{
				ade::CoordptrT revshaper(shaper->reverse());
				arg_shape = apply_shaper(revshaper, shape);
			}
			auto arg_rep = arg.arg_->make_rep(repr, owners, ctx, arg_shape);
			if (nullptr != arg_rep)
			{
				out_args.push_back(ReprArg<T>{arg_rep, shaper, arg.coorder_});
			}
		}
		for (size_t variadic : variadics_)
		{
			auto it = ctx.variadic_vars_.find(variadic);
			if (ctx.variadic_vars_.end() != it)
			{
				const RepArgsT<T>& variargs = it->second;
				out_args.insert(out_args.end(), variargs.begin(), variargs.end());
			}
		}
		if (out_args.empty())
		{
			// holds no value
			return RepptrT<T>(nullptr);
		}
		auto out = repr.make_func(opcode_, out_args);
		owners.emplace(out.get(), out);
		return out;
	}

	ade::Opcode opcode_;

	TargetArgsT<T> args_;

	std::vector<size_t> variadics_;
};

template <typename T>
struct RuleConversion final
{
	std::string str_form_;

	RuleptrT<T> source_;

	RuleTargetptrT<T> target_;
};

template <typename T>
using ConversionsT = std::vector<RuleConversion<T>>;

using UniqueTensT = std::unordered_map<std::string,ade::iTensor*>;

template <typename T>
void apply_rules (Representer<T>& repr, UniqueTensT& unique_tens,
	const ConversionsT<T>& conversions)
{
	OwnerMapT<T> ownermap;
	std::unordered_map<RepptrT<T>,std::vector<ade::iTensor*>> reverse_rep;
	std::list<RepptrT<T>> bottom_up;
	{
		size_t rule_minheight = 0;
		{
			std::vector<size_t> rule_minheights;
			rule_minheights.reserve(conversions.size());
			std::transform(conversions.begin(), conversions.end(),
				std::back_inserter(rule_minheights),
				[](const RuleConversion<T>& rule_convs)
				{
					return rule_convs.source_->get_minheight();
				});
			if (false == rule_minheights.empty())
			{
				rule_minheight = *std::min_element(
					rule_minheights.begin(), rule_minheights.end());
			}
		}

		std::unordered_map<RepptrT<T>,NumRange<size_t>> heights;
		for (auto reppair : repr.reps_)
		{
			auto& tens = reppair.first;
			auto& rep = reppair.second;
			NumRange<size_t> repheights = rep->get_heights();
			heights.emplace(rep, repheights);
			reverse_rep[rep].push_back(tens);
			ownermap.emplace(rep.get(), rep);
		}
		for (auto reppair : heights)
		{
			if (reppair.second.lower_ >= rule_minheight)
			{
				bottom_up.push_back(reppair.first);
			}
		}
		bottom_up.sort(
			[&](RepptrT<T>& a, RepptrT<T>& b)
			{
				return heights.at(a).upper_ < heights.at(b).upper_;
			});
	}

	ConversionsT<T> valid_conversions;
	valid_conversions.reserve(conversions.size());
	std::copy_if(conversions.begin(), conversions.end(),
		std::back_inserter(valid_conversions),
		[](const RuleConversion<T>& conv)
		{
			return nullptr != conv.target_ && nullptr != conv.source_;
		});
	if (valid_conversions.size() < conversions.size())
	{
		logs::error("applying rules with in invalid conversions");
	}

	for (RepptrT<T> rep : bottom_up)
	{
		auto& tens_keys = reverse_rep[rep];
		// there must be at least 1 tens mapped to tens in representer
		assert(tens_keys.size() > 0);
		ade::Shape exshape = tens_keys[0]->shape();
		std::string orig_id = rep->get_identifier();
		for (const RuleConversion<T>& conv : valid_conversions)
		{
			RuleContext<T> ctx;
			if (rep->rulify(ctx, conv.source_))
			{
				// match found, convert
				logs::debugf("applying conversion %s", conv.str_form_.c_str());
				rep = conv.target_->make_rep(repr, ownermap, ctx, exshape);
				if (nullptr == rep)
				{
					std::vector<T> zeros(exshape.n_elems(), 0);
					rep = std::make_shared<ConstRep<T>>(zeros.data(), exshape);
				}
			}
		}
		// update repr after conversion
		std::string fresh_id = rep->get_identifier();
		if (orig_id != fresh_id)
		{
			unique_tens.erase(orig_id);
			auto it = unique_tens.find(fresh_id);
			if (unique_tens.end() == it)
			{
				// no existing imprint, associate new rep
				unique_tens.emplace(fresh_id, tens_keys[0]);
			}
			else
			{
				// found existing imprint of rep, use existing rep instead
				rep = repr.reps_[it->second];
			}
			for (auto tens : tens_keys)
			{
				repr.replace_rep(tens, rep);
			}
		}
	}
}

// Optimize nodes while attempting to preserve edges using conversions
template <typename T>
void optimize (NodesT<T>& nodes, const ConversionsT<T>& conversions)
{
	Representer<T> repr;
	for (NodeptrT<T>& node : nodes)
	{
		auto tens = node->get_tensor();
		tens->accept(repr);
	}

	// map identifier to rep to reuse nodes
	UniqueTensT unique_tens;
	for (auto& reppair : repr.reps_)
	{
		auto& tens = reppair.first;
		auto& rep = reppair.second;
		std::string identifier = rep->get_identifier();
		auto it = unique_tens.find(identifier);
		if (unique_tens.end() == it)
		{
			unique_tens.emplace(identifier, tens);
		}
		else
		{
			repr.replace_rep(tens, repr.reps_[it->second]);
		}
	}

	apply_rules(repr, unique_tens, conversions);

	nodes = unrepresent(repr, nodes);
	nodes = ops_prune(nodes);
}

}

}

#endif // EAD_CONVERSION_HPP
