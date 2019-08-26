#include "pbm/save.hpp"
#include "pbm/load.hpp"

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

std::unordered_map<std::string,LayerId> unpack_labels (
	const std::vector<std::string>& labels)
{
	std::unordered_map<std::string,LayerId> out;
	for (std::string label : labels)
	{
		auto parts = fmts::split(label, std::string(&llabel_sep, 1));
		if (parts.size() < 4)
		{
			logs::errorf("invalid layer label %s", label.c_str());
			continue;
		}
		out.emplace(parts[0], LayerId(parts[1], parts[2], std::stoul(parts[3])));
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

using SublayersT = std::unordered_map<ade::iTensor*,std::list<LayerId>>;

struct LayerDeserializer final : public ade::OnceTraveler
{
	LayerDeserializer (std::string key, std::string val) :
		key_(key), val_(val) {}

	/// Implementation of OnceTraveler
	void visit_leaf (ade::iLeaf* leaf) override
	{
		tag::TagRepsT reps =
			tag::get_reg().get_tags(leaf);
		std::vector<std::string> labels;
		if (estd::get(labels, reps, key_))
		{
			auto layer_info = unpack_labels(labels);
			LayerId subid;
			if (estd::get(subid, layer_info, val_))
			{
				roots_.emplace(leaf);
				// gather all sublayer info
				std::list<LayerId> sub_info;
				while (false == subid.type_.empty())
				{
					sub_info.push_back(subid);
					if (key_ == subid.type_)
					{
						subid = estd::must_getf(layer_info, subid.label_,
							"cannot find label `%s` in leaf `%s` layer registry",
							subid.label_.c_str(), leaf->to_string().c_str());
					}
					else
					{
						auto sublabels = estd::must_getf(reps, subid.type_,
							"cannot find type `%s` in leaf `%s` layer registry",
							subid.type_.c_str(), leaf->to_string().c_str());
						auto sublayer_info = unpack_labels(sublabels);
						subid = estd::must_getf(sublayer_info, subid.label_,
							"cannot find label `%s` in leaf `%s` in %s",
							subid.label_.c_str(), leaf->to_string().c_str(),
							fmts::to_string(
								sublabels.begin(), sublabels.end()).c_str());
					}
				}
				layer_leaves_.emplace(leaf, sub_info);
			}
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
		if (estd::get(labels, reps, key_))
		{
			auto layer_info = unpack_labels(labels);
			LayerId subid;
			if (estd::get(subid, layer_info, val_))
			{
				for (auto child : children)
				{
					roots_.erase(child.get_tensor().get());
				}
				roots_.emplace(func);
			}
		}
	}

	SublayersT layer_leaves_;

	std::unordered_set<ade::iTensor*> roots_;

private:
	std::string key_;

	std::string val_;
};

void populate_builder (LBuilderptrT& builder,
	const ade::OwnerMapT& owners, const SublayersT& sublayers,
	LayerRegistry& registry)
{
	if (sublayers.empty())
	{
		return;
	}
	if (nullptr == builder)
	{
		logs::fatal("cannot populate null builder");
	}

	// group sublayers
	std::vector<std::pair<LBuilderptrT,SublayersT>> grouped_layers;
	for (auto llpair : sublayers)
	{
		if (llpair.second.empty())
		{
			// not a sublayer
			builder->set_tensor(owners.at(llpair.first).lock());
		}
		else
		{
			// is a sublayer
			auto id = llpair.second.front();
			llpair.second.pop_front();
			size_t n = grouped_layers.size();
			if (id.index_ >= n)
			{
				grouped_layers.insert(grouped_layers.end(), id.index_ - n + 1,
					std::pair<LBuilderptrT,SublayersT>{nullptr, SublayersT()});
			}
			if (nullptr == grouped_layers[id.index_].first)
			{
				grouped_layers[id.index_].first =
					registry.get_builder(id.type_)(id.label_);
			}
			grouped_layers[id.index_].second.emplace(
				llpair.first, llpair.second);
		}
	}
	for (auto& grouped_layer : grouped_layers)
	{
		populate_builder(grouped_layer.first, owners,
			grouped_layer.second, registry);
		builder->set_sublayer(grouped_layer.first->build());
	}
}

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
	auto root_builder = registry.get_builder(ltype)(label);
	populate_builder(root_builder, owners, layd.layer_leaves_, registry);
	return root_builder->build();
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
