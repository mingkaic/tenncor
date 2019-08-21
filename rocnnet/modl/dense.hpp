#include "ead/generated/api.hpp"

#include "rocnnet/eqns/init.hpp"

#include "rocnnet/modl/layer.hpp"
#include "rocnnet/modl/marshal.hpp"

#ifndef MODL_DENSE_HPP
#define MODL_DENSE_HPP

namespace modl
{

struct iLayer : public iMarshalSet
{
	iLayer (std::string label) :
		iMarshalSet(label) {}

	virtual ~iLayer (void) = default;

    iLayer* clone (void) const
    {
        return static_cast<iLayer*>(this->clone_impl());
    }

	virtual ead::NodeptrT<PybindT> connect (ead::NodeptrT<PybindT> input) const = 0;
};

using LayerptrT = std::shared_ptr<iLayer>;

const std::string dense_type = "dense";

struct Dense final : public iLayer
{
	Dense (ade::DimT nunits, ade::DimT indim,
		NonLinearF activation,
		eqns::InitF<PybindT> weight_init,
		eqns::InitF<PybindT> bias_init,
		std::string label) :
		iLayer(label),
		weight_(std::make_shared<MarshalVar>(
			weight_init(ade::Shape({nunits, indim}), label + ":weight"))),
		activation_(activation)
	{
		tag_layer(weight_->var_->get_tensor(), dense_type, label);
		if (bias_init)
		{
			bias_ = std::make_shared<MarshalVar>(
				bias_init(ade::Shape({nunits}), label + ":bias"));
			tag_layer(bias_->var_->get_tensor(), dense_type, label);
		}
	}

	Dense (ead::VarptrT<PybindT> weight, ead::VarptrT<PybindT> bias,
		NonLinearF activation, std::string label) :
		iLayer(label),
        weight_(std::make_shared<MarshalVar>(weight)),
        bias_(std::make_shared<MarshalVar>(bias)) {}

	Dense (const Dense& other) :
		iLayer(other)
	{
		copy_helper(other);
	}

	Dense& operator = (const Dense& other)
	{
		if (this != &other)
		{
			copy_helper(other);
		}
		return *this;
	}

	Dense (Dense&& other) = default;

	Dense& operator = (Dense&& other) = default;

    Dense* clone (void) const
    {
        return static_cast<Dense*>(this->clone_impl());
    }

	ead::NodeptrT<PybindT> connect (ead::NodeptrT<PybindT> input) const override
	{
		auto out = tenncor::nn::fully_connect({input},
			{ead::convert_to_node(weight_->var_)},
			ead::convert_to_node(bias_->var_));
		if (activation_)
		{
			out = activation_(out);
		}
		recursive_layer_tag(out->get_tensor(), dense_type, get_label(), {
			input->get_tensor().get(),
			weight_->var_->get_tensor().get(),
			bias_->var_->get_tensor().get(),
		});
		return out;
	}

	MarsarrT get_subs (void) const override
	{
		return MarsarrT{weight_, bias_};
	}

	NonLinearF activation_;

private:
	iMarshaler* clone_impl (void) const override
	{
		return new Dense(*this);
	}

	void copy_helper (const Dense& other)
	{
		weight_ = std::make_shared<MarshalVar>(*other.weight_),
		bias_ = std::make_shared<MarshalVar>(*other.bias_),
		activation_ = other.activation_;
	}

	MarVarsptrT weight_;

	MarVarsptrT bias_;
};

}

#endif // MODL_DENSE_HPP
