#include "estd/estd.hpp"

#include "tag/tag.hpp"

#include "eteq/constant.hpp"
#include "eteq/variable.hpp"

#include "eteq/generated/pyapi.hpp"

#ifndef LAYR_LAYER_HPP
#define LAYR_LAYER_HPP

namespace layr
{

const std::string layers_key_prefix = "layer_";

const char llabel_sep = ':';

void validate_label (const std::string& label);

struct LayerId final
{
	LayerId (void) = default;

	LayerId (std::string label) : label_(label) {}

	LayerId (std::string type, std::string label, size_t index) :
		type_(type), label_(label), index_(index) {}

	std::string to_string (std::string label) const
	{
		return fmts::sprintf("%s%c%s%c%s%c%d",
			label.c_str(), llabel_sep,
			type_.c_str(), llabel_sep,
			label_.c_str(), llabel_sep,
			index_);
	}

	std::string type_;

	std::string label_;

	size_t index_ = 0;
};

using LayerIdsT = std::vector<LayerId>;

std::string layer_label_fmt (std::string label, LayerId subid);

std::unordered_map<std::string,LayerIdsT> unpack_labels (
	const std::vector<std::string>& labels);

struct LayerTag final : public tag::iTag
{
	LayerTag (std::string layer_type, std::string name) :
		reps_({{layer_type, {name}}}) {}

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

struct iLayer
{
	virtual ~iLayer (void) = default;

	iLayer* clone (std::string label_prefix = "") const
	{
		return this->clone_impl(label_prefix);
	}

	virtual size_t get_ninput (void) const = 0;

	virtual size_t get_noutput (void) const = 0;

	virtual std::string get_ltype (void) const = 0;

	virtual std::string get_label (void) const = 0;

	virtual NodeptrT connect (
		NodeptrT input) const = 0;

	virtual teq::TensptrsT get_contents (void) const = 0;

protected:
	virtual iLayer* clone_impl (const std::string& label_prefix) const = 0;

	void tag (teq::TensptrT tensor, LayerId subs) const;

	void recursive_tag (teq::TensptrT root,
		teq::TensSetT ignores, LayerId subs) const;
};

using LayerptrT = std::shared_ptr<iLayer>;

struct iLayerBuilder
{
	virtual ~iLayerBuilder (void) = default;

	virtual void set_tensor (teq::TensptrT tens, std::string target) = 0;

	virtual void set_sublayer (LayerptrT layer) = 0;

	virtual LayerptrT build (void) const = 0;
};

using LBuilderptrT = std::shared_ptr<iLayerBuilder>;

using LayerBuildF = std::function<LBuilderptrT(std::string)>;

struct LayerRegistry final
{
	LayerRegistry (tag::TagRegistry& registry = tag::get_reg()) : tag_reg_(registry) {}

	void layer_tag (teq::TensrefT tens, std::string layer_type, std::string name)
	{
		tag_reg_.add_tag(tens, tag::TagptrT(new LayerTag(layer_type, name)));
	}

	std::string register_tagr (std::string key, tag::TagrF tagr, LayerBuildF builder)
	{
		lbuilders_.emplace(key, builder);

		return tag_reg_.register_tagr(key,
		[this, key](teq::TensrefT ref, std::string label)
		{
			this->layer_tag(ref, key, label);
		});
	}

	LayerBuildF get_builder (std::string layer_type)
	{
		return estd::must_getf(lbuilders_, layer_type,
			"failed to find registered layer `%s`", layer_type.c_str());
	}

	tag::TagRegistry& get_tag_registry (void)
	{
		return tag_reg_;
	}

private:
	std::unordered_map<std::string,LayerBuildF> lbuilders_;

	tag::TagRegistry& tag_reg_;
};

LayerRegistry& get_layer_reg (void);

void recursive_layer_tag (teq::TensptrT tens, std::string layer_type,
	std::string name, teq::TensSetT stops,
	LayerRegistry& registry = get_layer_reg());

LayerptrT load_layer (std::istream& ins, teq::TensptrsT& roots,
	std::string ltype, std::string label,
	LayerRegistry& registry = get_layer_reg());

bool save_layer (std::ostream& outs, const iLayer& layer, teq::TensptrsT roots,
	LayerRegistry& registry = get_layer_reg());

}

#endif // LAYR_LAYER_HPP
