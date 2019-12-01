///
/// seqmodel.hpp
/// layr
///
/// Purpose:
/// Implement sequentially connected model
///

#include "layr/dense.hpp"

#ifndef LAYR_SEQMODEL_HPP
#define LAYR_SEQMODEL_HPP

namespace layr
{

/// Builder implementation for sequentially connected models
struct SeqModelBuilder final : public iLayerBuilder
{
	SeqModelBuilder (std::string label) : label_(label) {}

	/// Implementation of iLayerBuilder
	void set_tensor (teq::TensptrT tens, std::string target) override {} // seqmodel has no tensor

	/// Implementation of iLayerBuilder
	void set_sublayer (LayerptrT layer) override
	{
		layers_.push_back(layer);
	}

	/// Implementation of iLayerBuilder
	LayerptrT build (void) const override;

private:
	std::string label_;

	std::vector<LayerptrT> layers_;
};

/// Identifier for sequentially connected models
const std::string seq_model_key =
get_layer_reg().register_tagr(layers_key_prefix + "seqmodel",
[](std::string label) -> LBuilderptrT
{
	return std::make_shared<SeqModelBuilder>(label);
});

/// Layer implementation that sequentially applies sublayers
struct SequentialModel final : public iLayer
{
	SequentialModel (const std::string& label) :
		label_(label) {}

	SequentialModel (const SequentialModel& other,
		std::string label_prefix = "")
	{
		copy_helper(other, label_prefix);
	}

	SequentialModel& operator = (const SequentialModel& other)
	{
		if (this != &other)
		{
			copy_helper(other);
		}
		return *this;
	}

	SequentialModel (SequentialModel&& other) = default;

	SequentialModel& operator = (SequentialModel&& other) = default;

	/// Return deep copy of this model with prefixed label
	SequentialModel* clone (std::string label_prefix = "") const
	{
		return static_cast<SequentialModel*>(this->clone_impl(label_prefix));
	}

	/// Implementation of iLayer
	size_t get_ninput (void) const override
	{
		size_t input = 0;
		for (auto it = layers_.begin(), et = layers_.end();
			it != et && 0 == input; ++it)
		{
			input = (*it)->get_ninput();
		}
		return input;
	}

	/// Implementation of iLayer
	size_t get_noutput (void) const override
	{
		size_t output = 0;
		for (auto it = layers_.rbegin(), et = layers_.rend();
			it != et && 0 == output; ++it)
		{
			output = (*it)->get_noutput();
		}
		return output;
	}

	/// Implementation of iLayer
	std::string get_ltype (void) const override
	{
		return seq_model_key;
	}

	/// Implementation of iLayer
	std::string get_label (void) const override
	{
		return label_;
	}

	/// Implementation of iLayer
	teq::TensptrsT get_contents (void) const override
	{
		teq::TensptrsT out;
		out.reserve(layers_.size());
		for (auto& layer : layers_)
		{
			auto tmp = layer->get_contents();
			out.insert(out.end(), tmp.begin(), tmp.end());
		}
		return out;
	}

	/// Implementation of iLayer
	NodeptrT connect (NodeptrT input) const override
	{
		NodeptrT output;
		for (size_t i = 0, n = layers_.size(); i < n; ++i)
		{
			auto& layer = layers_[i];
			output = layer->connect(input);
			recursive_tag(output->get_tensor(), {
				input->get_tensor().get()
			}, LayerId(layer->get_ltype(), layer->get_label(), i));
			input = output;
		}
		return output;
	}

	/// Return stored sublayers
	std::vector<LayerptrT> get_layers (void) const
	{
		return layers_;
	}

	/// Append layer to stored sublayers
	void push_back (LayerptrT layer)
	{
		// label layer content
		auto subs = layer->get_contents();
		for (auto& sub : subs)
		{
			if (nullptr != sub)
			{
				tag(sub, LayerId(layer->get_ltype(),
					layer->get_label(), layers_.size()));
			}
		}

		layers_.push_back(layer);
	}

private:
	iLayer* clone_impl (const std::string& label_prefix) const override
	{
		return new SequentialModel(*this, label_prefix);
	}

	void copy_helper (const SequentialModel& other, std::string label_prefix = "")
	{
		label_ = label_prefix + other.label_;
		layers_.clear();
		layers_.reserve(other.layers_.size());
		for (LayerptrT olayer : other.layers_)
		{
			push_back(LayerptrT(olayer->clone(label_prefix)));
		}
	}

	std::string label_;

	std::vector<LayerptrT> layers_;
};

/// Smart pointer of sequentially connected model
using SeqModelptrT = std::shared_ptr<SequentialModel>;

}

#endif // LAYR_SEQMODEL_HPP
