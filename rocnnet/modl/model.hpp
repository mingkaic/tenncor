#include "rocnnet/modl/dense.hpp"

#ifndef MODL_MODEL_HPP
#define MODL_MODEL_HPP

namespace modl
{

struct SeqModelBuilder final : public iLayerBuilder
{
	SeqModelBuilder (std::string label) : label_(label) {}

	void set_tensor (ade::TensptrT tens) override {} // seqmodel has no tensor

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
[](ade::TensrefT ref, std::string label)
{
	get_layer_reg().layer_tag(ref, seq_model_key, label);
},
[](std::string label) -> LBuilderptrT
{
	return std::make_shared<SeqModelBuilder>(label);
});

struct SequentialModel final : public iLayer
{
	SequentialModel (std::string label) :
		label_(label) {}

	SequentialModel (const SequentialModel& other)
	{
		copy_helper(other);
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

	SequentialModel* clone (void) const
	{
		return static_cast<SequentialModel*>(this->clone_impl());
	}

	std::string get_ltype (void) const override
	{
		return seq_model_key;
	}

	std::string get_label (void) const override
	{
		return label_;
	}

	ead::NodeptrT<PybindT> connect (ead::NodeptrT<PybindT> input) const override
	{
		ead::NodeptrT<PybindT> out;
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

	ade::TensT get_contents (void) const override
	{
		ade::TensT out;
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
	iLayer* clone_impl (void) const override
	{
		return new SequentialModel(*this);
	}

	void copy_helper (const SequentialModel& other)
	{
		label_ = other.label_;
		layers_.clear();
		layers_.reserve(other.layers_.size());
		for (LayerptrT olayer : other.layers_)
		{
			push_back(olayer);
		}
	}

	std::string label_;

	std::vector<LayerptrT> layers_;
};

using SeqModelptrT = std::shared_ptr<SequentialModel>;

}

#endif // MODL_MODEL_HPP
