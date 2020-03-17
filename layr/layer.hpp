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
		return RBMLayer<T>{eteq::deep_clone(fwd_), eteq::deep_clone(bwd_)};
	}

	eteq::ETensor<T> connect (const eteq::ETensor<T>& input) const
	{
		return eteq::connect(fwd_, input);
	}

	eteq::ETensor<T> backward_connect (const eteq::ETensor<T>& output) const
	{
		return eteq::connect(bwd_, output);
	}

	eteq::ETensor<T> fwd_;

	eteq::ETensor<T> bwd_;
};

}

#endif // LAYR_LAYER_HPP
