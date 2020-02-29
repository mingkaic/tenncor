///
/// api.hpp
/// layr
///
/// Purpose:
/// Utility APIs for creating layers
///

#include "eteq/layer.hpp"
#include "eteq/serialize.hpp"

#include "layr/init.hpp"

#ifndef LAYR_LAYER_HPP
#define LAYR_LAYER_HPP

namespace layr
{

const std::string weight_label = "weight";

const std::string bias_label = "bias";

const std::string input_label = "input";

teq::Shape gen_rshape (std::vector<teq::DimT> runcoms,
	teq::Shape left, eigen::PairVecT<teq::RankT> lrdims);

template <typename T>
using UnaryF = std::function<eteq::ETensor<T>(eteq::ETensor<T>)>;

template <typename T>
struct RBMLayer final
{
	RBMLayer<T> deep_clone (void) const
	{
		return RBMLayer<T>{fwd_.deep_clone(), bwd_.deep_clone()};
	}

	eteq::ELayer<T> fwd_;

	eteq::ELayer<T> bwd_;
};

template <typename T>
eteq::ELayer<T> bind (UnaryF<T> unary, teq::Shape inshape = teq::Shape())
{
	eteq::ETensor<T> input(eteq::make_variable_scalar<T>(
		0, inshape, input_label));
	auto output = unary(input);
	return eteq::ELayer<T>(
		estd::must_ptr_cast<teq::iFunctor>((teq::TensptrT) output), input);
}

template <typename T>
eteq::ELayer<T> link (const std::vector<eteq::ELayer<T>>& layers,
	const eteq::ETensor<T>& input)
{
	if (layers.empty())
	{
		teq::fatal("cannot link no layers");
	}
	auto output = input;
	for (size_t i = 0, n = layers.size(); i < n; ++i)
	{
		auto& lay = layers[i];
		if (lay.input().get() == input.get())
		{
			output = eteq::ETensor<T>(lay.root());
		}
		else
		{
			output = lay.connect(output);
		}
	}
	return eteq::ELayer<T>(std::static_pointer_cast<
		teq::iFunctor>(teq::TensptrT(output)), input);
}

template <typename T>
eteq::ELayer<T> link (const std::vector<eteq::ELayer<T>>& layers)
{
	if (layers.empty())
	{
		teq::fatal("cannot link no layers");
	}
	eteq::ETensor<T> input = layers.front().input();
	return link<T>(layers, input);
}

}

#endif // LAYR_LAYER_HPP
