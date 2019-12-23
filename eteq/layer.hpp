#include "eteq/functor.hpp"
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
struct Trailer final : public teq::OnceTraveler
{
	Trailer (const teq::TensMapT<teq::TensptrT>& inputs) :
		trailed_(inputs), pfinder_(replace_targets(inputs)) {}

	teq::TensMapT<teq::TensptrT> trailed_;

private:
	/// Implementation of OnceTraveler
	void visit_leaf (teq::iLeaf& leaf) override {}

	/// Implementation of OnceTraveler
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
		for (std::string attr : attr_directions)
		{
			auto ref = static_cast<const teq::TensorRef*>(func.get_attr(attr));
			auto ctens = ref->get_tensor();
			ctens->accept(*this);
			dup_attrs.rm_attr(attr);
			dup_attrs.add_attr(attr, marsh::ObjptrT(ref->copynreplace(
				trailed_.at(ctens.get()))));
		}

		auto& child_directions = target_dir.children_;
		teq::TensptrsT children = func.get_children();
		for (size_t i : child_directions)
		{
			auto child = children[i];
			child->accept(*this);
			children[i] = trailed_.at(child.get());
		}

		trailed_.emplace(&func, Functor<T>::get(
			(egen::_GENERATED_OPCODE) func.get_opcode().code_,
			children, std::move(dup_attrs)));
	}

	teq::PathFinder pfinder_;
};

/// Copy everything from input.first to root, except replacing input.first with input.second
template <typename T>
ETensor<T> trail (const ETensor<T>& root,
	const teq::TensMapT<teq::TensptrT>& inputs)
{
	Trailer<T> trailer(inputs);
	root->accept(trailer);
	return estd::try_get(trailer.trailed_, root.get(), nullptr);
}

template <typename T>
void get_storage (VarptrsT<T>& storages, const teq::TensptrT& root)
{
	if (auto f = dynamic_cast<const teq::iFunctor*>(root.get()))
	{
		if (auto lattr = f->get_attr(teq::layer_key))
		{
			auto layer = static_cast<const teq::LayerObj*>(lattr);
			auto input = layer->get_tensor();

			// find all variables between root and input
			teq::GraphStat stats;
			stats.graphsize_.emplace(input.get(), estd::NumRange<size_t>());
			root->accept(stats);
			teq::OwnerMapT owner = teq::track_owners({root});

			for (auto gpair : stats.graphsize_)
			{
				if (0 == gpair.second.upper_ && input.get() != gpair.first)
				{
					if (auto var = std::dynamic_pointer_cast<
						Variable<T>>(owner.at(gpair.first).lock()))
					{
						storages.push_back(var);
					}
				}
			}
		}
		auto children = f->get_children();
		for (auto child : children)
		{
			get_storage(storages, ETensor<T>(child));
		}
	}
}

template <typename T>
struct ELayer final
{
	ELayer (teq::FuncptrT root, ETensor<T> input) :
		root_(root), input_(input) {}

	ELayer<T> deep_clone (void) const
	{
		auto oinput = input_;
		teq::Copier kamino({input_.get()});
		root_->accept(kamino);
		auto oroot = kamino.clones_.at(root_.get());
		auto f = std::static_pointer_cast<teq::iFunctor>(
			(teq::TensptrT) oroot);
		return ELayer<T>(f, oinput);
	}

	ETensor<T> connect (const ETensor<T>& oinput) const
	{
		return trail(ETensor<T>(root_), teq::TensMapT<teq::TensptrT>{{input_.get(), (teq::TensptrT) oinput}});
	}

	teq::FuncptrT root (void) const
	{
		return root_;
	}

	ETensor<T> input (void) const
	{
		return input_;
	}

	VarptrsT<T> get_storage (void) const
	{
		VarptrsT<T> vars;
		::eteq::get_storage<T>(vars, root_);
		return vars;
	}

private:
	teq::FuncptrT root_;

	ETensor<T> input_;
};

const std::string input_key = "model_input";

template <typename T>
void save_layer (onnx::ModelProto& model, const ELayer<T>& layer)
{
	onnx::TensIdT ids;
	ids.insert({layer.input().get(), input_key});
	save_model(model, {layer.root()}, ids);
}

template <typename T>
ELayer<T> load_layer (const onnx::ModelProto& model)
{
	onnx::TensptrIdT ids;
	auto roots = load_model(ids, model);
	if (roots.empty())
	{
		logs::fatal("cannot load model without roots");
	}
	auto input = estd::must_getf(ids.right, input_key,
		"failed to find %s", input_key.c_str());
	auto root = roots.front();
	auto f = std::static_pointer_cast<teq::iFunctor>(
		(teq::TensptrT) root);
	return ELayer<T>(f, ETensor<T>(input));
}

}

#endif // ETEQ_LAYER_HPP
