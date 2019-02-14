#include <functional>
#include <memory>

#include "rocnnet/modl/fcon.hpp"

#ifndef MODL_MLP_HPP
#define MODL_MLP_HPP

namespace modl
{

using HiddenFunc = std::function<ead::NodeptrT<double>(ead::NodeptrT<double>)>;

const std::string hidden_fmt = "hidden_%d";

struct LayerInfo
{
	size_t n_out_;
	HiddenFunc hidden_;
};

struct MLP final : public iMarshalSet
{
	MLP (uint8_t n_input, std::vector<LayerInfo> layers, std::string label) :
		iMarshalSet(label)
	{
		for (size_t i = 0, n = layers.size(); i < n; ++i)
		{
			size_t n_output = layers[i].n_out_;
			layers_.push_back(HiddenLayer{
				std::make_shared<FCon>(std::vector<uint8_t>{n_input},
					n_output, fmts::sprintf(hidden_fmt, i)),
				layers[i].hidden_
			});
			n_input = n_output;
		}
	}

	MLP (const MLP& other) : iMarshalSet(other)
	{
		copy_helper(other);
	}

	MLP& operator = (const MLP& other)
	{
		if (this != &other)
		{
			iMarshalSet::operator = (other);
			copy_helper(other);
		}
		return *this;
	}

	MLP (MLP&& other) = default;

	MLP& operator = (MLP&& other) = default;


	ead::NodeptrT<double> operator () (ead::NodeptrT<double> input)
	{
		ead::NodeptrT<double> out = input;
		for (HiddenLayer& layer : layers_)
		{
			auto hypothesis = (*layer.layer_)({out});
			out = layer.hidden_(hypothesis);
		}
		return out;
	}

	uint8_t get_ninput (void) const
	{
		return layers_.front().layer_->get_ninput();
	}

	uint8_t get_noutput (void) const
	{
		return layers_.back().layer_->get_noutput();
	}

	MarsarrT get_subs (void) const override
	{
		MarsarrT out(layers_.size());
		std::transform(layers_.begin(), layers_.end(), out.begin(),
			[](const HiddenLayer& layer)
			{
				return layer.layer_;
			});
		return out;
	}

private:
	void copy_helper (const MLP& other)
	{
		layers_.clear();
		for (const HiddenLayer& olayer : other.layers_)
		{
			layers_.push_back(HiddenLayer{
				std::make_shared<FCon>(*olayer.layer_),
				olayer.hidden_
			});
		}
	}

	iMarshaler* clone_impl (void) const override
	{
		return new MLP(*this);
	}

	struct HiddenLayer
	{
		FConptrT layer_;
		HiddenFunc hidden_;
	};

	std::vector<HiddenLayer> layers_;
};

using MLPptrT = std::shared_ptr<MLP>;

}

#endif // MODL_MLP_HPP
