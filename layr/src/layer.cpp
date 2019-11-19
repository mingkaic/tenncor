#include "pbm/save.hpp"
#include "pbm/load.hpp"

#include "eteq/serialize.hpp"

#include "layr/layer.hpp"

#ifdef LAYR_LAYER_HPP

namespace layr
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

void iLayer::tag (teq::TensptrT tensor, LayerId subs) const
{
	// todo: allow pass layer registry through parameter
	get_layer_reg().layer_tag(tensor, get_ltype(),
		subs.to_string(get_label()));
}

void iLayer::recursive_tag (teq::TensptrT root,
	teq::TensSetT ignores, LayerId subs) const
{
	// todo: allow pass layer registry through parameter
	recursive_layer_tag(root, get_ltype(),
		subs.to_string(get_label()), ignores);
}

LayerRegistry& get_layer_reg (void)
{
	static LayerRegistry registry;
	return registry;
}

void recursive_layer_tag (teq::TensptrT tens, std::string layer_type,
	std::string name, teq::TensSetT stops,
	LayerRegistry& registry)
{
	tag::recursive_tag(tens, stops,
		[&](teq::TensrefT ref)
		{
			registry.layer_tag(ref, layer_type, name);
		});
}

struct LayerNode;

using LNodeptrT = std::shared_ptr<LayerNode>;

using LMatchesT = std::list<std::pair<LayerNode*,std::string>>;

struct LayerNode final
{
	LayerNode (const std::string& type, const std::string& label) :
		type_(type), label_(label) {}

	// Return generated leaves after adding branches to the subtree from tag reps
	LMatchesT match_layer (const tag::TagRepsT& reps)
	{
		LMatchesT matches;
		std::vector<std::string> labels;
		if (estd::get(labels, reps, type_))
		{
			auto layer_info = unpack_labels(labels);
			LayerIdsT subids;
			if (estd::get(subids, layer_info, label_))
			{
				for (LayerId subid : subids)
				{
					if (subid.type_.empty())
					{
						matches.push_back({this, subid.label_});
					}
					else
					{
						size_t n = subs_.size();
						if (subid.index_ >= n)
						{
							subs_.insert(subs_.end(),
								subid.index_ - n + 1, LNodeptrT());
						}
						auto& sub = subs_[subid.index_];
						if (nullptr == sub)
						{
							sub = std::make_shared<LayerNode>(
								subid.type_, subid.label_);
						}
						auto tmp = sub->match_layer(reps);
						matches.insert(matches.end(), tmp.begin(), tmp.end());
					}
				}
			}
		}
		return matches;
	}

	std::string type_;

	std::string label_;

	std::vector<LNodeptrT> subs_;
};

using SublayersT = std::unordered_map<teq::iTensor*,LMatchesT>;

using TensLablT = std::unordered_map<teq::TensptrT,std::string>;

using TensLayerMapT = std::unordered_map<LayerNode*,TensLablT>;

struct LayerDeserializer final : public teq::OnceTraveler
{
	LayerDeserializer (std::string key, std::string val) :
		base_(std::make_shared<LayerNode>(key, val)) {}

	/// Implementation of OnceTraveler
	void visit_leaf (teq::iLeaf* leaf) override
	{
		tag::TagRepsT reps =
			tag::get_reg().get_tags(leaf);
		LMatchesT matches = base_->match_layer(reps);
		if (false == matches.empty())
		{
			sublayers_.emplace(leaf, matches);
			roots_.emplace(leaf);
		}
	}

	/// Implementation of OnceTraveler
	void visit_func (teq::iFunctor* func) override
	{
		auto children = func->get_children();
		for (const teq::iEdge& child : children)
		{
			child.get_tensor()->accept(*this);
		}

		tag::TagRepsT reps =
			tag::get_reg().get_tags(func);
		LMatchesT matches = base_->match_layer(reps);
		if (false == matches.empty())
		{
			sublayers_.emplace(func, matches);
			for (const teq::iEdge& child : children)
			{
				roots_.erase(child.get_tensor().get());
			}
			roots_.emplace(func);
		}
	}

	LayerptrT build_layer (
		LayerRegistry& registry, teq::OwnerMapT& owners) const
	{
		TensLayerMapT layer_tens;
		for (auto& sublayer : sublayers_)
		{
			auto& tens = sublayer.first;
			auto& matches = sublayer.second;
			for (auto& match : matches)
			{
				layer_tens[match.first].emplace(
					owners.at(tens).lock(), match.second);
			}
		}
		return build_layer_helper(registry, layer_tens, base_.get());
	}

	teq::TensSetT roots_;

private:
	LayerptrT build_layer_helper (LayerRegistry& registry,
		const TensLayerMapT& layer_tens, LayerNode* lroot) const
	{
		if (nullptr == lroot)
		{
			logs::fatal("cannot builder layer from null layer node");
		}
		auto builder = registry.get_builder(lroot->type_)(lroot->label_);
		for (const LNodeptrT& sub : lroot->subs_)
		{
			if (nullptr != sub)
			{
				builder->set_sublayer(build_layer_helper(
					registry, layer_tens, sub.get()));
			}
		}
		TensLablT tenslabels;
		if (estd::get(tenslabels, layer_tens, lroot))
		{
			for (auto& tenslabel : tenslabels)
			{
				builder->set_tensor(tenslabel.first, tenslabel.second);
			}
		}
		return builder->build();
	}

	SublayersT sublayers_;

	LNodeptrT base_;
};

LayerptrT load_layer (std::istream& ins, teq::TensptrsT& roots,
	std::string ltype, std::string label,
	LayerRegistry& registry)
{
	tenncor::Graph graph;
	if (false == graph.ParseFromIstream(&ins))
	{
		logs::fatalf("failed to parse from istream when loading %s",
			ltype.c_str());
	}
	teq::TensptrSetT info;
	eteq::load_graph(info, graph);

	teq::OwnerMapT owners = teq::track_owners(
		teq::TensptrsT(info.begin(), info.end()));

	LayerDeserializer layd(ltype, label);
	// get all layer labelled nodes in graph
	for (teq::TensptrT tens : info)
	{
		tens->accept(layd);
	}

	roots.reserve(layd.roots_.size());
	for (teq::iTensor* root : layd.roots_)
	{
		roots.push_back(owners.at(root).lock());
	}

	return layd.build_layer(registry, owners);
}

bool save_layer (std::ostream& outs, const iLayer& layer, teq::TensptrsT roots,
	LayerRegistry& registry)
{
	auto contents = layer.get_contents();
	roots.insert(roots.end(), contents.begin(), contents.end());

	// save graph from source
	tenncor::Graph graph;
	eteq::save_graph(graph, roots, registry.get_tag_registry());
	return graph.SerializeToOstream(&outs);
}

}

#endif // LAYR_LAYER_HPP
