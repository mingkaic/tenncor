#include "pbm/save.hpp"
#include "pbm/load.hpp"

#include "ead/serialize.hpp"

#include "rocnnet/modl/layer.hpp"

#ifdef MODL_LAYER_HPP

namespace modl
{

size_t LayerTag::tag_id_ = typeid(LayerTag).hash_code();

void validate_label (const std::string& label)
{
	if (std::string::npos != label.find(llabel_sep))
	{
		logs::fatalf("label `%s` cannot have reserved separator '%c'",
			label.c_str(), llabel_sep);
	}
}

std::unordered_map<std::string,LayerIdsT> unpack_labels (
	const std::vector<std::string>& labels)
{
	std::unordered_map<std::string,LayerIdsT> out;
	for (std::string label : labels)
	{
		auto parts = fmts::split(label, std::string(&llabel_sep, 1));
		if (parts.size() < 4)
		{
			logs::errorf("invalid layer label %s", label.c_str());
			continue;
		}
		out[parts[0]].push_back(LayerId(parts[1], parts[2], std::stoul(parts[3])));
	}
	return out;
}

void iLayer::tag (ade::TensptrT tensor, LayerId subs) const
{
	get_layer_reg().layer_tag(tensor, get_ltype(),
		subs.to_string(get_label()));
}

void iLayer::recursive_tag (ade::TensptrT root,
	std::unordered_set<ade::iTensor*> ignores, LayerId subs) const
{
	recursive_layer_tag(root, get_ltype(),
		subs.to_string(get_label()), ignores);
}

LayerRegistry& get_layer_reg (void)
{
	static LayerRegistry registry;
	return registry;
}

void recursive_layer_tag (ade::TensptrT tens, std::string layer_type,
	std::string name, std::unordered_set<ade::iTensor*> stops,
	LayerRegistry& registry)
{
	tag::recursive_tag(tens, stops,
		[&](ade::TensrefT ref)
		{
			registry.layer_tag(ref, layer_type, name);
		});
}

struct LayerNode;

using LNodeptrT = std::shared_ptr<LayerNode>;

struct LayerNode final
{
	LayerNode (const std::string& type, const std::string& label) :
		type_(type), label_(label) {}

	// Return generated leaves after adding branches to the subtree from tag reps
	std::list<LNodeptrT> match_layer (const tag::TagRepsT& reps)
	{
		std::list<LNodeptrT> matches;
		std::vector<std::string> labels;
		if (estd::get(labels, reps, type_))
		{
			auto layer_info = unpack_labels(labels);
			LayerIdsT subids;
			if (estd::get(subids, layer_info, label_))
			{
				for (LayerId subid : subids)
				{
					size_t n = subs_.size();
					if (subid.index_ >= n)
					{
						subs_.insert(subs_.end(), subid.index_ - n + 1, LNodeptrT());
					}
					auto& sub = subs_[subid.index_];
					if (nullptr == sub)
					{
						sub = std::make_shared<LayerNode>(subid.type_, subid.label_);
					}
					auto tmp = sub->match_layer(reps);
					matches.insert(matches.end(), tmp.begin(), tmp.end());
				}
			}
		}
		return matches;
	}

	std::string type_;

	std::string label_;

	std::vector<LNodeptrT> subs_;
};

using SublayersT = std::unordered_map<ade::iTensor*,std::list<LNodeptrT>>;

using TensLayerMapT = std::unordered_map<LNodeptrT,ade::TensT>;

struct LayerDeserializer final : public ade::OnceTraveler
{
	LayerDeserializer (std::string key, std::string val) :
		base_(std::make_shared<LayerNode>(key, val)) {}

	/// Implementation of OnceTraveler
	void visit_leaf (ade::iLeaf* leaf) override
	{
		tag::TagRepsT reps =
			tag::get_reg().get_tags(leaf);
		std::list<LNodeptrT> matches = base_->match_layer(reps);
		if (false == matches.empty())
		{
			layer_leaves_.emplace(leaf, matches);
		}
	}

	/// Implementation of OnceTraveler
	void visit_func (ade::iFunctor* func) override
	{
		auto& children = func->get_children();
		for (auto child : children)
		{
			child.get_tensor()->accept(*this);
		}

		tag::TagRepsT reps =
			tag::get_reg().get_tags(func);
		std::vector<std::string> labels;
		if (estd::get(labels, reps, base_->type_))
		{
			auto layer_info = unpack_labels(labels);
			if (estd::has(layer_info, base_->label_))
			{
				for (auto child : children)
				{
					roots_.erase(child.get_tensor().get());
				}
				roots_.emplace(func);
			}
		}
	}

	LayerptrT build_layer (
		LayerRegistry& registry, ade::OwnerMapT& owners) const
	{
		TensLayerMapT layer_tens;
		for (auto& layer_leaf : layer_leaves_)
		{
			auto& tens = layer_leaf.first;
			auto& nodes = layer_leaf.second;
			for (auto& node : nodes)
			{
				layer_tens[node].push_back(owners.at(tens).lock());
			}
		}
		return build_layer_helper(registry, layer_tens, base_);
	}

	std::unordered_set<ade::iTensor*> roots_;

private:
	LayerptrT build_layer_helper (LayerRegistry& registry,
		const TensLayerMapT& layer_tens, const LNodeptrT& lroot) const
	{
		auto builder = registry.get_builder(lroot->type_)(lroot->label_);
		for (const LNodeptrT& sub : lroot->subs_)
		{
			builder->set_sublayer(build_layer_helper(
				registry, layer_tens, sub));
		}
		ade::TensT tensors;
		if (estd::get(tensors, layer_tens, lroot))
		{
			for (auto& tens : tensors)
			{
				builder->set_tensor(tens);
			}
		}
		return builder->build();
	}

	SublayersT layer_leaves_;

	LNodeptrT base_;
};

LayerptrT load_layer (std::istream& ins, ade::TensT& roots,
	std::string ltype, std::string label,
	LayerRegistry& registry)
{
	cortenn::Graph graph;
	if (false == graph.ParseFromIstream(&ins))
	{
		logs::fatalf("failed to parse from istream when loading %s",
			ltype.c_str());
	}
	pbm::GraphInfo info;
	pbm::load_graph<ead::EADLoader>(info, graph);

	ade::OwnerMapT owners = ade::track_owners(
		ade::TensT(info.roots_.begin(), info.roots_.end()));

	LayerDeserializer layd(ltype, label);
	// get all layer labelled nodes in graph
	for (ade::TensptrT tens : info.roots_)
	{
		tens->accept(layd);
	}

	roots.reserve(layd.roots_.size());
	for (ade::iTensor* root : layd.roots_)
	{
		roots.push_back(owners.at(root).lock());
	}

	return layd.build_layer(registry, owners);
}

bool save_layer (std::ostream& outs, const iLayer& layer, ade::TensT roots,
	LayerRegistry& registry)
{
	pbm::GraphSaver<ead::EADSaver> saver(registry.get_tag_registry());
	for (auto& root : roots)
	{
		root->accept(saver);
	}

	auto contents = layer.get_contents();
	auto owners = ade::track_owners(contents);
	for (auto tens : contents)
	{
		tens->accept(saver);
	}

	pbm::PathedMapT labels;
	for (ade::iLeaf* leaf : saver.leaves_)
	{
		if (false == leaf->is_const())
		{
			labels.emplace(owners.at(leaf).lock(), pbm::StringsT{leaf->to_string()});
		}
	}

	// save graph from source
	cortenn::Graph graph;
	saver.save(graph, labels);
	return graph.SerializeToOstream(&outs);
}

}

#endif // MODL_LAYER_HPP
