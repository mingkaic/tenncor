///
/// api.hpp
/// layr
///
/// Purpose:
/// Utility APIs for creating layers
///

#ifndef LAYR_LAYER_HPP
#define LAYR_LAYER_HPP

#include "tenncor/layr/init.hpp"

namespace layr
{

const std::string weight_label = "weight";

const std::string bias_label = "bias";

const std::string input_label = "input";

const std::string bind_name = "_UNARY_BIND";

const std::string link_name = "_LINK";

const std::string dense_name = "_DENSE_LAYER";

const std::string conv_name = "_CONV_LAYER";

const std::string rnn_name = "_RNN_LAYER";

const std::string lstm_name = "_LSTM_LAYER";

const std::string gru_name = "_GRU_LAYER";

static inline teq::TensMapT<std::string> replace_targets (
	const teq::OwnMapT& inputs)
{
	teq::TensMapT<std::string> targets;
	for (auto& inp : inputs)
	{
		targets.emplace(inp.first, "target");
	}
	return targets;
}

teq::Shape gen_rshape (teq::DimsT runcoms,
	teq::Shape left, eigen::PairVecT<teq::RankT> lrdims);

teq::TensptrT make_layer (teq::TensptrT root,
	const std::string& layername, teq::TensptrT input);

eteq::ETensor get_input (const eteq::ETensor& root);

/// Copy everything from input.first to root, except replacing input.first with input.second
eteq::ETensor trail (const eteq::ETensor& root, const teq::OwnMapT& inputs);

eteq::ETensor connect (const eteq::ETensor& root, const eteq::ETensor& input);

eteq::ETensor deep_clone (const eteq::ETensor& root);

struct Trailer final : public teq::iOnceTraveler
{
	Trailer (const teq::OwnMapT& inputs) :
		trailed_(inputs), pfinder_(replace_targets(inputs)) {}

	teq::OwnMapT trailed_;

private:
	/// Implementation of iOnceTraveler
	void visit_leaf (teq::iLeaf&) override {}

	/// Implementation of iOnceTraveler
	void visit_func (teq::iFunctor& func) override
	{
		if (estd::has(trailed_, &func))
		{
			return;
		}
		func.accept(pfinder_);
		if (false == estd::has(pfinder_.roadmap_, &func))
		{
			return;
		}
		auto& target_dir = pfinder_.roadmap_.at(&func).at("target");

		marsh::Maps dup_attrs;
		marsh::get_attrs(dup_attrs, func);
		auto& attr_directions = target_dir.attrs_;
		for (const std::string& attr : attr_directions)
		{
			auto ref = static_cast<const teq::TensorRef*>(func.get_attr(attr));
			auto ctens = ref->get_tensor();
			ctens->accept(*this);
			teq::TensptrT trailed;
			if (estd::get(trailed, trailed_, ctens.get()))
			{
				dup_attrs.rm_attr(attr);
				dup_attrs.add_attr(attr, marsh::ObjptrT(
					ref->copynreplace(trailed)));
			}
		}

		auto& child_directions = target_dir.args_;
		teq::TensptrsT children = func.get_args();
		for (size_t i : child_directions)
		{
			auto child = children[i];
			child->accept(*this);
			children[i] = trailed_.at(child.get());
		}

		auto opcode = (egen::_GENERATED_OPCODE) func.get_opcode().code_;
		auto fcpy = eteq::make_funcattr(opcode, children, dup_attrs);
		trailed_.emplace(&func, fcpy);
	}

	teq::PathFinder pfinder_;
};

struct BreadthStat final : public teq::iOnceTraveler
{
	teq::TensMapT<size_t> breadth_;

private:
	void visit_leaf (teq::iLeaf& leaf) override
	{
		breadth_.emplace(&leaf, breadth_.size());
	}

	void visit_func (teq::iFunctor& func) override
	{
		auto deps = func.get_args();
		teq::multi_visit(*this, deps);
		breadth_.emplace(&func, breadth_.size());
	}
};

// Inorder traversal for mutable leaves
struct VarExtract final : public teq::iOnceTraveler
{
	VarExtract (teq::TensSetT term = {}) : term_(term) {}

	teq::LeafsT variables_;

	teq::TensSetT term_;

private:
	void visit_leaf (teq::iLeaf& leaf) override
	{
		if (estd::has(term_, &leaf))
		{
			return;
		}
		if (teq::Usage::IMMUTABLE != leaf.get_usage())
		{
			variables_.push_back(&leaf);
		}
	}

	void visit_func (teq::iFunctor& func) override
	{
		if (estd::has(term_, &func))
		{
			return;
		}
		auto deps = func.get_args();
		teq::multi_visit(*this, deps);
	}
};

template <typename T>
eteq::VarptrsT<T> get_storage (const eteq::ETensor& root)
{
	teq::RefMapT owner = teq::track_ownrefs(teq::TensptrsT{root});

	auto intens = get_input(root).get();
	VarExtract extra({intens});
	root->accept(extra);

	eteq::VarptrsT<T> vars;
	vars.reserve(extra.variables_.size());
	for (auto leaf : extra.variables_)
	{
		if (auto var = std::dynamic_pointer_cast<
			eteq::Variable<T>>(owner.at(leaf).lock()))
		{
			vars.push_back(var);
		}
	}
	return vars;
}

template <typename T>
struct RBMLayer final
{
	RBMLayer<T> deep_clone (void) const
	{
		return RBMLayer<T>{
			::layr::deep_clone(fwd_),
			::layr::deep_clone(bwd_)
		};
	}

	eteq::ETensor connect (const eteq::ETensor& input) const
	{
		return ::layr::connect(fwd_, input);
	}

	eteq::ETensor backward_connect (const eteq::ETensor& output) const
	{
		return ::layr::connect(bwd_, output);
	}

	eteq::ETensor fwd_;

	eteq::ETensor bwd_;
};

using UnaryF = std::function<eteq::ETensor(const eteq::ETensor&)>;

}

#endif // LAYR_LAYER_HPP
