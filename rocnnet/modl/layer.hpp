#include "tag/tag.hpp"

#ifndef MODL_LAYER_HPP
#define MODL_LAYER_HPP

namespace modl
{

const std::string layers_key_prefix = "layer_";

struct LayerTag final : public tag::iTag
{
	LayerTag (std::string layer_type, std::string name) :
		reps_({{layers_key_prefix + layer_type, {name}}}) {}

	size_t tag_id (void) const override
	{
		return tag_id_;
	}

	void absorb (tag::TagptrT&& other) override
	{
		LayersT& oreps =
			static_cast<LayerTag*>(other.get())->reps_;
		for (auto& reppair : oreps)
		{
			auto& names = reps_[reppair.first];
			names.insert(reppair.second.begin(), reppair.second.end());
		}
	}

	tag::TagRepsT get_tags (void) const override
	{
        tag::TagRepsT out;
        for (auto& layer : reps_)
        {
            out.emplace(layer.first, std::vector<std::string>(
                layer.second.begin(), layer.second.end()));
        }
		return out;
	}

private:
    using LayersT = std::map<std::string,std::unordered_set<std::string>>;

	LayersT reps_;

	static size_t tag_id_;
};

void tag_layer (ade::TensrefT tens, std::string layer_type, std::string name,
	tag::TagRegistry& registry = tag::get_reg());

void recursive_layer_tag (ade::TensptrT tens, std::string layer_type,
    std::string name, std::unordered_set<ade::iTensor*> stops,
	tag::TagRegistry& registry = tag::get_reg());

}

#endif // MODL_LAYER_HPP
