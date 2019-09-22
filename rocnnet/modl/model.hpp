#include "rocnnet/modl/dense.hpp"

#ifndef MODL_MODEL_HPP
#define MODL_MODEL_HPP

namespace modl
{

struct SeqModelBuilder final : public iLayerBuilder
{
	SeqModelBuilder (std::string label) : label_(label) {}

	void set_tensor (teq::TensptrT tens, std::string target) override {} // seqmodel has no tensor

	void set_sublayer (LayerptrT layer) override
	{
		layers_.push_back(layer);
	}

	LayerptrT build (void) const override;

private:
	std::string label_;

	std::vector<LayerptrT> layers_;
};

const std::string seq_model_key =
get_layer_reg().register_tagr(layers_key_prefix + "seqmodel",
[](teq::TensrefT ref, std::string label)
{
	get_layer_reg().layer_tag(ref, seq_model_key, label);
},
[](std::string label) -> LBuilderptrT
{
	return std::make_shared<SeqModelBuilder>(label);
});

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

	SequentialModel* clone (std::string label_prefix = "") const
	{
		return static_cast<SequentialModel*>(this->clone_impl(label_prefix));
	}

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

	std::string get_ltype (void) const override
	{
		return seq_model_key;
	}

	std::string get_label (void) const override
	{
		return label_;
	}

	eteq::NodeptrT<PybindT> connect (eteq::NodeptrT<PybindT> input) const override
	{
		eteq::NodeptrT<PybindT> out;
		for (size_t i = 0, n = layers_.size(); i < n; ++i)
		{
			auto& layer = layers_[i];
			out = layer->connect(input);
			input = out;
			recursive_tag(out->get_tensor(), {
				input->get_tensor().get(),
			}, LayerId(layer->get_ltype(), layer->get_label(), i));
		}
		return out;
	}

	teq::TensT get_contents (void) const override
	{
		teq::TensT out;
		out.reserve(layers_.size());
		for (auto& layer : layers_)
		{
			auto tmp = layer->get_contents();
			out.insert(out.end(), tmp.begin(), tmp.end());
		}
		return out;
	}

	void push_back (LayerptrT layer)
	{
		// label layer content
		auto subs = layer->get_contents();
		for (auto& sub : subs)
		{
			tag(sub, LayerId(layer->get_ltype(),
				layer->get_label(), layers_.size()));
		}

		layers_.push_back(layer);
	}

private:
	iLayer* clone_impl (std::string label_prefix) const override
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

using SeqModelptrT = std::shared_ptr<SequentialModel>;

}

#endif // MODL_MODEL_HPP
