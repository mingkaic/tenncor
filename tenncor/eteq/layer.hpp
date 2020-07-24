#include "eteq/serialize.hpp"

#ifndef ETEQ_LAYER_HPP
#define ETEQ_LAYER_HPP

namespace eteq
{

static inline teq::TensMapT<std::string> replace_targets (
	const teq::TensMapT<teq::TensptrT>& inputs)
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
	Trailer (const teq::TensMapT<teq::TensptrT>& inputs) :
		trailed_(inputs), pfinder_(replace_targets(inputs)) {}

	teq::TensMapT<teq::TensptrT> trailed_;

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

		trailed_.emplace(&func, make_funcattr<T>(
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
ETensor<T> trail (const ETensor<T>& root,
	const teq::TensMapT<teq::TensptrT>& inputs)
{
	Trailer<T> trailer(inputs);
	root->accept(trailer);
	return ETensor<T>(estd::try_get(trailer.trailed_, root.get(), nullptr),
		root.get_context());
}

template <typename T>
ETensor<T> get_input (const ETensor<T>& root)
{
	if (nullptr == root)
	{
		teq::fatal("cannot get layer attr with null root");
	}
	auto froot = estd::must_ptr_cast<teq::iFunctor>((teq::TensptrT) root);
	auto layerattr = estd::must_cast<teq::LayerObj>(froot->get_attr(teq::layer_key));
	return ETensor<T>(layerattr->get_tensor(), root.get_context());
}

template <typename T>
ETensor<T> connect (const ETensor<T>& root, const ETensor<T>& input)
{
	return trail(root, teq::TensMapT<teq::TensptrT>{
		{get_input(root).get(), (teq::TensptrT) input}});
}

template <typename T>
VarptrsT<T> get_storage (const ETensor<T>& root)
{
	teq::OwnerMapT owner = teq::track_owners({teq::TensptrT(root)});

	auto intens = get_input(root).get();
	VarExtract extra({intens});
	root->accept(extra);

	VarptrsT<T> vars;
	vars.reserve(extra.variables_.size());
	for (auto leaf : extra.variables_)
	{
		if (auto var = std::dynamic_pointer_cast<
			Variable<T>>(owner.at(leaf).lock()))
		{
			vars.push_back(var);
		}
	}
	return vars;
}

template <typename T>
ETensor<T> deep_clone (const ETensor<T>& root)
{
	teq::Copier kamino({get_input(root).get()});
	root->accept(kamino);
	return ETensor<T>(kamino.clones_.at(root.get()), root.get_context());
}

}

#endif // ETEQ_LAYER_HPP
