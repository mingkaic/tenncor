///
/// api.hpp
/// layr
///
/// Purpose:
/// Utility APIs for creating layers
///

#include "tenncor/layr/init.hpp"

#ifndef LAYR_LAYER_HPP
#define LAYR_LAYER_HPP

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

teq::Shape gen_rshape (std::vector<teq::DimT> runcoms,
	teq::Shape left, eigen::PairVecT<teq::RankT> lrdims);

teq::TensptrT make_layer (teq::TensptrT root,
	const std::string& layername, teq::TensptrT input);

template <typename T>
eteq::ETensor<T> get_input (const eteq::ETensor<T>& root)
{
	if (nullptr == root)
	{
		global::fatal("cannot get layer attr with null root");
	}
	auto froot = estd::must_ptr_cast<teq::iFunctor>((teq::TensptrT) root);
	auto layerattr = estd::must_cast<teq::LayerObj>(froot->get_attr(teq::layer_key));
	return eteq::ETensor<T>(layerattr->get_tensor(), root.get_context());
}

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

template <typename T>
struct Trailer final : public teq::iOnceTraveler
{
	Trailer (const teq::OwnMapT& inputs) :
		trailed_(inputs), pfinder_(replace_targets(inputs)) {}

	teq::OwnMapT trailed_;

private:
	/// Implementation of iOnceTraveler
	void visit_leaf (teq::iLeaf& leaf) override {}

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

		auto& child_directions = target_dir.children_;
		teq::TensptrsT children = func.get_dependencies();
		for (size_t i : child_directions)
		{
			auto child = children[i];
			child->accept(*this);
			children[i] = trailed_.at(child.get());
		}

		trailed_.emplace(&func, eteq::make_funcattr<T>(
			(egen::_GENERATED_OPCODE) func.get_opcode().code_,
			children, dup_attrs));
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
		auto deps = func.get_dependencies();
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
		auto deps = func.get_dependencies();
		teq::multi_visit(*this, deps);
	}
};

/// Copy everything from input.first to root, except replacing input.first with input.second
template <typename T>
eteq::ETensor<T> trail (const eteq::ETensor<T>& root,
	const teq::OwnMapT& inputs)
{
	Trailer<T> trailer(inputs);
	root->accept(trailer);
	return eteq::ETensor<T>(estd::try_get(trailer.trailed_, root.get(), nullptr),
		root.get_context());
}

template <typename T>
eteq::ETensor<T> connect (const eteq::ETensor<T>& root, const eteq::ETensor<T>& input)
{
	return trail(root, teq::OwnMapT{
		{get_input(root).get(), (teq::TensptrT) input}});
}

template <typename T>
eteq::VarptrsT<T> get_storage (const eteq::ETensor<T>& root)
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
eteq::ETensor<T> deep_clone (const eteq::ETensor<T>& root)
{
	teq::Copier kamino({get_input(root).get()});
	root->accept(kamino);
	return eteq::ETensor<T>(kamino.clones_.at(root.get()), root.get_context());
}

template <typename T>
using UnaryF = std::function<eteq::ETensor<T>(const eteq::ETensor<T>&)>;

template <typename T>
struct RBMLayer final
{
	RBMLayer<T> deep_clone (void) const
	{
		return RBMLayer<T>{
			::layr::deep_clone<T>(fwd_),
			::layr::deep_clone<T>(bwd_)
		};
	}

	eteq::ETensor<T> connect (const eteq::ETensor<T>& input) const
	{
		return ::layr::connect<T>(fwd_, input);
	}

	eteq::ETensor<T> backward_connect (const eteq::ETensor<T>& output) const
	{
		return ::layr::connect<T>(bwd_, output);
	}

	eteq::ETensor<T> fwd_;

	eteq::ETensor<T> bwd_;
};

}

#endif // LAYR_LAYER_HPP
