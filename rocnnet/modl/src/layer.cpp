#include "rocnnet/modl/layer.hpp"

#ifdef MODL_LAYER_HPP

namespace modl
{

size_t LayerTag::tag_id_ = typeid(LayerTag).hash_code();

void tag_layer (ade::TensrefT tens, std::string layer_type, std::string name,
	tag::TagRegistry& registry)
{
	registry.add_tag(tens, tag::TagptrT(new LayerTag(layer_type, name)));
}

void recursive_layer_tag (ade::TensptrT tens, std::string layer_type,
    std::string name, std::unordered_set<ade::iTensor*> stops,
	tag::TagRegistry& registry)
{
	tag::recursive_tag(tens, stops,
		[&](ade::TensrefT ref)
		{
            tag_layer(ref, layer_type, name, registry);
		});
}

}

#endif // MODL_LAYER_HPP
