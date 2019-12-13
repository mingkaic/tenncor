///
/// layer.hpp
/// layr
///
/// Purpose:
/// Define layer interface and tagging
///

#include "estd/estd.hpp"

#include "eteq/generated/pyapi.hpp"

#include "eteq/make.hpp"
#include "eteq/placeholder.hpp"

#ifndef LAYR_LAYER_HPP
#define LAYR_LAYER_HPP

namespace layr
{

/// String prefixed to every layer key
const std::string layers_key_prefix = "layer_";

/// Layer label separator to divide each element in the LayerId
const char llabel_sep = ':';

/// Check if the raw label does not contain llabel_sep
/// as to not clash with LayerId representation
void validate_label (const std::string& label);

/// Sublayer type, label, and index encapsulation
struct LayerId final
{
	LayerId (void) = default;

	LayerId (std::string label) : label_(label) {}

	LayerId (std::string type, std::string label, size_t index) :
		type_(type), label_(label), index_(index) {}

	/// Represent layer by the following format
	/// <raw label>:<sublayer type>:<sublayer label>:<sublayer index>
	std::string to_string (std::string label) const
	{
		return fmts::sprintf("%2$s%1$c%3$s%1$c%4$s%1$c%5$d",
			llabel_sep,
			label.c_str(),
			type_.c_str(),
			label_.c_str(),
			index_);
	}

	/// Sublayer type
	std::string type_;

	/// Sublayer label
	std::string label_;

	/// Sublayer index
	size_t index_ = 0;
};

/// Vector of sublayer ids
using LayerIdsT = std::vector<LayerId>;

/// Return formatted raw label with associated sublayer
std::string layer_label_fmt (std::string label, LayerId subid);

/// Return raw labels mapped to sublayers given a vector of formatted labels
std::unordered_map<std::string,LayerIdsT> unpack_labels (
	const std::vector<std::string>& labels);

/// Tag implementation specifically for contents of layers
struct LayerTag final : public tag::iTag
{
	LayerTag (std::string layer_type, std::string name) :
		reps_({{layer_type, {name}}}) {}

	/// Implementation of iTag
	size_t tag_id (void) const override
	{
		return tag_id_;
	}

	/// Implementation of iTag
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

	/// Implementation of iTag
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

/// Layer interface defining io, content and metadata getters
/// as well as the connector
struct iLayer
{
	virtual ~iLayer (void) = default;

	/// Return deep copy of this layer with prefixed label
	iLayer* clone (std::string label_prefix = "") const
	{
		return this->clone_impl(label_prefix);
	}

	// todo: remove this in favor of get_input_sign
	/// Return input value of the expected input (first dimension)
	virtual size_t get_ninput (void) const = 0;

	// todo: remove this in favor of get_output_sign
	/// Return output value of the expected output (first dimension)
	virtual size_t get_noutput (void) const = 0;

	/// Return input value of the expected input (first dimension)
	teq::ShapeSignature get_input_sign (void) const
	{
		return input_->shape_sign();
	}

	/// Return output value of the expected output (first dimension)
	teq::ShapeSignature get_output_sign (void) const
	{
		return output_->shape_sign();
	}

	/// Return the layer type which is also the tag key of tagged contents
	virtual std::string get_ltype (void) const = 0;

	/// Return the raw layer label
	virtual std::string get_label (void) const = 0;

	/// Return all internal tensors representing the layer
	virtual teq::TensptrsT get_contents (void) const = 0;

	/// Return the root of the graph that connects input with internal tensors
	virtual LinkptrT connect (LinkptrT input) const = 0;

protected:
	void placeholder_connect (const teq::ShapeSignature& insign)
	{
		if (nullptr == input_)
		{
			input_ = std::make_shared<eteq::PlaceLink<PybindT>>(
				insign, this->get_label() + "_input");
		}
		else
		{
			assert(input_->shape_sign().compatible_after(insign, 0));
		}
		output_ = this->connect(input_);
	}

	virtual iLayer* clone_impl (const std::string& label_prefix) const = 0;

	void tag (teq::TensptrT tensor, LayerId subs) const;

	eteq::PlaceLinkptrT<PybindT> input_ = nullptr;

	LinkptrT output_ = nullptr;
};

/// Smart pointer of layer
using LayerptrT = std::shared_ptr<iLayer>;

/// Layer builder interface defining internal tensor
/// and sublayer setting and layer building
/// (like a poor-man's dependency injector interface)
struct iLayerBuilder
{
	virtual ~iLayerBuilder (void) = default;

	/// Set internal tensors that make up the output layer
	virtual void set_tensor (teq::TensptrT tens, std::string target) = 0;

	/// Set internal sublayers that make up the output layer
	virtual void set_sublayer (LayerptrT layer) = 0;

	/// Return the layer built to contain set tensors and sublayers
	virtual LayerptrT build (void) const = 0;
};

/// Layer builder smart pointer
using LBuilderptrT = std::shared_ptr<iLayerBuilder>;

/// Function that takes layer type and returns associated layer builder
using LayerBuildF = std::function<LBuilderptrT(std::string)>;

/// Registry object for associating layer type and layer builders as well as
/// registering layer type as tag keys
struct LayerRegistry final
{
	LayerRegistry (tag::TagRegistry& registry = tag::get_reg()) : tag_reg_(registry) {}

	/// Tag tens reference with layer type and label
	void layer_tag (teq::TensrefT tens, std::string layer_type, std::string name)
	{
		tag_reg_.add_tag(tens, tag::TagptrT(new LayerTag(layer_type, name)));
	}

	/// Return key (layer type) that is associated with builder
	/// and registered in tag registry
	std::string register_tagr (std::string key, LayerBuildF builder)
	{
		lbuilders_.emplace(key, builder); // todo: remove tagr since it's not used

		return tag_reg_.register_tagr(key,
		[this, key](teq::TensrefT ref, std::string label)
		{
			this->layer_tag(ref, key, label);
		});
	}

	/// Return builder associated with layer type
	LayerBuildF get_builder (std::string layer_type)
	{
		return estd::must_getf(lbuilders_, layer_type,
			"failed to find registered layer `%s`", layer_type.c_str());
	}

	/// Return wrapped tag registry refence
	tag::TagRegistry& get_tag_registry (void)
	{
		return tag_reg_;
	}

private:
	std::unordered_map<std::string,LayerBuildF> lbuilders_;

	tag::TagRegistry& tag_reg_;
};

/// Return global layer registry reference
LayerRegistry& get_layer_reg (void);

/// Recursively tag tensor subgraph with specified layer type, and label
/// only ignoring subgraphs of tensors in stops set
void recursive_layer_tag (teq::TensptrT tens, std::string layer_type,
	std::string name, teq::TensSetT stops,
	LayerRegistry& registry = get_layer_reg());

/// Return a rebuilt layer from protobuf in stream (ins) a bunch of subgraph roots
/// and the output layer's type and label
LayerptrT load_layer (std::istream& ins, teq::TensptrsT& roots,
	std::string ltype, std::string label,
	LayerRegistry& registry = get_layer_reg());

/// Return true if specified layer and root subgraphs are
/// saved to protobuf out stream (outs)
bool save_layer (std::ostream& outs, const iLayer& layer, teq::TensptrsT roots,
	LayerRegistry& registry = get_layer_reg());

}

#endif // LAYR_LAYER_HPP
