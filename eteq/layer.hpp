#include "estd/cast.hpp"

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
			auto attrval = func.get_attr(attr);
			if (auto ref = dynamic_cast<const teq::TensorObj*>(attrval))
			{
				auto ctens = ref->get_tensor();
				ctens->accept(*this);
				teq::TensptrT trailed;
				if (estd::get(trailed, trailed_, ctens.get()))
				{
					dup_attrs.rm_attr(attr);
					dup_attrs.add_attr(attr,
						marsh::ObjptrT(ref->copynreplace(trailed)));
				}
			}
			else if (auto lattr = dynamic_cast<const teq::LayerArrayT*>(attrval))
			{
				dup_attrs.rm_attr(attr);
				dup_attrs.add_attr(attr, std::make_unique<teq::LayerArrayT>());
				auto& cplayers = static_cast<teq::LayerArrayT*>(
					dup_attrs.get_attr(attr))->contents_;
				lattr->foreach(
					[&](size_t i, const marsh::iObject* obj)
					{
						auto layer = estd::must_cast<const teq::LayerObj>(obj);
						auto ctens = layer->get_tensor();
						ctens->accept(*this);
						teq::TensptrT trailed;
						if (estd::get(trailed, this->trailed_, ctens.get()))
						{
							cplayers.insert(cplayers.end(),
								teq::LayerptrT(static_cast<teq::LayerObj*>(
									layer->copynreplace(trailed))));
						}
						else
						{
							cplayers.insert(cplayers.end(),
								teq::LayerptrT(layer->clone()));
						}
					});
			}
		}

		auto& child_directions = target_dir.children_;
		teq::TensptrsT children = func.get_children();
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
		auto children = func.get_children();
		for (teq::TensptrT child : children)
		{
			child->accept(*this);
		}
		breadth_.emplace(&func, breadth_.size());
	}
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
std::vector<std::string> ls_layerattrs (const ETensor<T>& root)
{
	auto froot = estd::must_ptr_cast<teq::iFunctor>((teq::TensptrT) root);
	auto layers = estd::must_cast<teq::LayerArrayT>(
		froot->get_attr(teq::layers_key));

	std::vector<std::string> attrs;
	attrs.reserve(layers->contents_.size());
	layers->foreach(
		[&](size_t i, marsh::iObject* obj)
		{
			attrs.push_back(obj->to_string());
		});
	return attrs;
}

template <typename T>
teq::LayerObj* get_layerattr (
	const std::string& layername, const ETensor<T>& root)
{
	if (nullptr == root)
	{
		teq::fatalf("cannot get layer attr %s with null root",
			layername.c_str());
	}
	auto froot = estd::must_ptr_cast<teq::iFunctor>((teq::TensptrT) root);
	auto layers = estd::must_cast<marsh::iArray>(
		froot->get_attr(teq::layers_key));
	teq::LayerObj* layerattr = nullptr;
	layers->foreach(
		[&](size_t i, marsh::iObject* obj)
		{
			auto layer = estd::must_cast<teq::LayerObj>(obj);
			if (layer->to_string() == layername)
			{
				layerattr = layer;
			}
		});
	return layerattr;
}

template <typename T>
ETensor<T> get_input (const std::string& layername, const ETensor<T>& root)
{
	teq::LayerObj* layerattr = get_layerattr(layername, root);
	if (nullptr == layerattr)
	{
		teq::fatalf("cannot get input from %s without an layer attribute",
			layername.c_str());
	}
	return layerattr->get_tensor();
}

template <typename T>
ETensor<T> connect (const std::string& layername,
	const ETensor<T>& root, const ETensor<T>& input)
{
	return trail(root, teq::TensMapT<teq::TensptrT>{
		{get_input(layername, root).get(), (teq::TensptrT) input}});
}

template <typename T>
VarptrsT<T> get_storage (const std::string& layername, const ETensor<T>& root)
{
	auto intens = get_input(layername, root).get();
	VarptrsT<T> vars;

	teq::GraphStat stats;
	stats.graphsize_.emplace(intens, estd::NumRange<size_t>());
	root->accept(stats);

	BreadthStat bstats;
	bstats.breadth_.emplace(intens, 0);
	root->accept(bstats);

	teq::OwnerMapT owner = teq::track_owners({teq::TensptrT(root)});

	for (auto gpair : stats.graphsize_)
	{
		if (0 == gpair.second.upper_ && intens != gpair.first)
		{
			if (auto var = std::dynamic_pointer_cast<
				Variable<T>>(owner.at(gpair.first).lock()))
			{
				vars.push_back(var);
			}
		}
	}
	std::sort(vars.begin(), vars.end(),
		[&bstats](VarptrT<T> a, VarptrT<T> b)
		{
			return bstats.breadth_.at(a.get()) <
				bstats.breadth_.at(b.get());
		});
	return vars;
}

template <typename T>
ETensor<T> deep_clone (const std::string& layername, const ETensor<T>& root)
{
	teq::Copier kamino({get_input(layername, root).get()});
	root->accept(kamino);
	return kamino.clones_.at(root.get());
}

}

#endif // ETEQ_LAYER_HPP
