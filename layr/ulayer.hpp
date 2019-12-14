///
/// ulayer.hpp
/// layr
///
/// Purpose:
/// Implement generic layer that applies unary functions
/// these functions don't store any data
///

#include "teq/ilayer.hpp"

#include "eteq/generated/api.hpp"

#ifndef LAYR_ULAYER_HPP
#define LAYR_ULAYER_HPP

namespace layr
{

/// Function that takes corresponding unary layer and node
using UnaryF = std::function<LinkptrT(LinkptrT)>;

/// Layer implementation to apply activation and pooling functions
template <typename T=PybindT>
struct ULayer final : public teq::iLayer
{
	ULayer (teq::ShapeSignature insign,
		UnaryF func, const std::string& label = "") :
		iLayer(insign), label_(label),
		output_(func(eteq::to_link<T>(this->input_))) {}

	ULayer (const ULayer& other,
		std::string label_prefix = "")
	{
		copy_helper(other, label_prefix);
	}

	ULayer& operator = (const ULayer& other)
	{
		if (this != &other)
		{
			copy_helper(other);
		}
		return *this;
	}

	ULayer (ULayer&& other) = default;

	ULayer& operator = (ULayer&& other) = default;

	/// Return deep copy of this layer with prefixed label
	ULayer* clone (std::string label_prefix = "") const
	{
		return static_cast<ULayer*>(this->clone_impl(label_prefix));
	}

	/// Implementation of iLayer
	teq::ShapeSignature get_output_sign (void) const override
	{
		return output_->shape_sign();
	}

	/// Implementation of iLayer
	teq::TensptrT get_output (void) const override
	{
		return output_->get_tensor();
	}

	/// Implementation of iLayer
	std::string get_ltype (void) const override
	{
		return "unary_layer";
	}

	/// Implementation of iLayer
	std::string get_label (void) const override
	{
		return label_;
	}

	/// Implementation of iLayer
	teq::TensptrsT get_storage (void) const override
	{
		return {};
	}

	/// Implementation of iLayer
	teq::TensptrT connect (teq::TensptrT input) const override
	{
		input_->assign(eteq::to_link<T>(input)->build_data());
		auto output = eteq::data_link<T>(output_->build_data())->get_tensor();
		input_->clear();
		return output;
	}

private:
	iLayer* clone_impl (const std::string& label_prefix) const override
	{
		return new ULayer(*this, label_prefix);
	}

	void copy_helper (const ULayer& other, std::string label_prefix = "")
	{
		label_ = label_prefix + other.get_label();
		teq::Copier kamino({this->input_.get()});
		auto oout = other.output_->get_tensor();
		oout->accept(kamino);
		output_ = eteq::to_link<T>(kamino.clones_[oout.get()]);
	}

	std::string label_;

	eteq::LinkptrT<T> output_;
};

/// Smart pointer of unary layer
using UnaryptrT = std::shared_ptr<ULayer>;

/// Return activation layer using sigmoid
UnaryptrT sigmoid (std::string label = "");

/// Return activation layer using tanh
UnaryptrT tanh (std::string label = "");

/// Return activation layer using relu
UnaryptrT relu (std::string label = "");

/// Return activation layer using softmax of specified dimension
UnaryptrT softmax (teq::RankT dim, std::string label = "");

/// Return pooling layer using max aggregation
UnaryptrT maxpool2d (
	std::pair<teq::DimT,teq::DimT> dims = {0, 1},
	std::string label = "");

/// Return pooling layer using mean aggregation
UnaryptrT meanpool2d (
	std::pair<teq::DimT,teq::DimT> dims = {0, 1},
	std::string label = "");

}

#endif // LAYR_ULAYER_HPP
