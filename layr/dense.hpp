///
/// dense.hpp
/// layr
///
/// Purpose:
/// Implement fully connected layer
///

#include "teq/ilayer.hpp"

#include "eteq/generated/api.hpp"

#include "layr/init.hpp"

#ifndef LAYR_DENSE_HPP
#define LAYR_DENSE_HPP

namespace layr
{

/// Fully connected weight label
const std::string dense_weight_key = "weight";

/// Fully connected bias label
const std::string dense_bias_key = "bias";

teq::ShapeSignature matmul_leftsign (teq::Shape right,
	eigen::PairVecT<teq::RankT> dims) const
{
	std::vector<teq::DimT> slist(teq::rank_cap, 0);
	for (auto dpair : dims)
	{
		slist[dpair.first] = right.at(dpair.second);
	}
	return teq::ShapeSignature(slist);
}

teq::Shape prefix_shape (teq::DimT prefix, teq::Shape right)
{
	teq::Shape outshape({prefix});
	std::copy(right.begin(),
		right.begin() + teq::rank_cap - 1, outshape.begin() + 1);
	return outshape;
}

/// Layer implementation to apply fully_connect functions to weight and optional bias
template <typename T=PybindT>
struct Dense final : public teq::iLayer
{
	Dense (teq::DimT nunits, const teq::Shape& inshape,
		layr::InitF<T> weight_init, layr::InitF<T> bias_init,
		const std::string& label,
		eigen::PairVecT<teq::RankT> dims = {{0, 1}}) :
		iLayer(matmul_leftsign(prefix_shape(nunits, inshape), dims)),
		label_(label),
		weight_(weight_init(prefix_shape(nunits, inshape), dense_weight_key)),
		dims_(dims)
	{
		if (bias_init)
		{
			bias_ = bias_init(teq::Shape({nunits}), dense_bias_key);
		}
		output_ = connect(this->input_);
	}

	Dense (const Dense& other, std::string label_prefix = "") : iLayer(other)
	{
		copy_helper(other, label_prefix);
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

	/// Return deep copy of this layer with prefixed label
	Dense* clone (std::string label_prefix = "") const
	{
		return static_cast<Dense*>(this->clone_impl(label_prefix));
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
		return "dense";
	}

	/// Implementation of iLayer
	std::string get_label (void) const override
	{
		return label_;
	}

	/// Implementation of iLayer
	teq::TensptrsT get_storage (void) const override
	{
		return {weight_, bias_};
	}

	/// Implementation of iLayer
	teq::TensptrT connect (teq::TensptrT input) const override
	{
		// expect input to be <input dimensions...,nbatch dimensions...>
		return tenncor::nn::fully_connect({eteq::to_link<T>(input)},
			{eteq::to_link<T>(weight_)}, eteq::to_link<T>(bias_), dims_);
	}

private:
	iLayer* clone_impl (const std::string& label_prefix) const override
	{
		return new Dense(*this, label_prefix);
	}

	void copy_helper (const Dense& other, std::string label_prefix = "")
	{
		label_ = label_prefix + other.label_;
		weight_ = teq::TensptrT(other.weight_->clone());
		if (other.bias_)
		{
			bias_ = teq::TensptrT(other.bias_->clone());
		}
		dims_ = other.dims_;
		output_ = connect(this->input_);
	}

	std::string label_;

	teq::TensptrT weight_;

	teq::TensptrT bias_ = nullptr;

	eigen::PairVecT<teq::RankT> dims_;

	eteq::LinkptrT<T> output_;
};

/// Smart pointer of fully connected layer
using DenseptrT = std::shared_ptr<Dense>;

}

#endif // LAYR_DENSE_HPP
