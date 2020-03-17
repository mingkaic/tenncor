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

const std::string bind_name = "_UNARY_BIND";

const std::string link_name = "_LINK";

const std::string dense_name = "_DENSE_LAYER";

const std::string conv_name = "_CONV_LAYER";

const std::string rnn_name = "_RNN_LAYER";

const std::string lstm_name = "_LSTM_LAYER";

const std::string gru_name = "_GRU_LAYER";

teq::Shape gen_rshape (std::vector<teq::DimT> runcoms,
	teq::Shape left, eigen::PairVecT<teq::RankT> lrdims);

template <typename T>
using UnaryF = std::function<eteq::ETensor<T>(const eteq::ETensor<T>&)>;

template <typename T>
struct RBMLayer final
{
	RBMLayer<T> deep_clone (void) const
	{
		return RBMLayer<T>{eteq::deep_clone(dense_name, fwd_), eteq::deep_clone(dense_name, bwd_)};
	}

	eteq::ETensor<T> connect (const eteq::ETensor<T>& input) const
	{
		return eteq::connect(dense_name, fwd_, input);
	}

	eteq::ETensor<T> backward_connect (const eteq::ETensor<T>& output) const
	{
		return eteq::connect(dense_name, bwd_, output);
	}

	eteq::ETensor<T> fwd_;

	eteq::ETensor<T> bwd_;
};

template <typename T>
eteq::ETensor<T> bind (UnaryF<T> unary, teq::Shape inshape = teq::Shape())
{
	eteq::ETensor<T> input(eteq::make_variable_scalar<T>(
		0, inshape, input_label));
	auto output = unary(input);
	auto f = std::static_pointer_cast<teq::iFunctor>((teq::TensptrT) output);
	output = eteq::make_layer<T>(bind_name, input, f);
	return output;
}

template <typename T>
using LayersT = std::vector<std::pair<std::string,eteq::ETensor<T>>>;

template <typename T>
eteq::ETensor<T> link (const LayersT<T>& layers,
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
		if (eteq::get_layerattr(lay.first, lay.second)->
			get_tensor().get() == output.get())
		{
			output = lay.second;
		}
		else
		{
			output = eteq::connect(lay.first, lay.second, output);
		}
	}
	auto f = std::static_pointer_cast<teq::iFunctor>((teq::TensptrT) output);
	output = eteq::make_layer<T>(link_name, input, f);
	return output;
}

template <typename T>
eteq::ETensor<T> link (const LayersT<T>& layers)
{
	if (layers.empty())
	{
		teq::fatal("cannot link no layers");
	}
	std::string firstlayer = layers.front().first;
	teq::LayerObj* layerattr = get_layerattr(firstlayer, layers.front().second);
	if (nullptr == layerattr)
	{
		teq::fatalf("cannot get_storage from %s without no layer attribute",
			firstlayer.c_str());
	}
	return link<T>(layers, layerattr->get_tensor());
}

}

#endif // LAYR_LAYER_HPP
