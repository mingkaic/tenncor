///
/// dense.hpp
/// layr
///
/// Purpose:
/// Implement fully connected layer
///

#include "eteq/generated/api.hpp"
#include "eteq/layer.hpp"

#include "layr/init.hpp"

#ifndef LAYR_API_HPP
#define LAYR_API_HPP

namespace layr
{

/// Fully connected weight label
const std::string weight_key = "weight";

/// Fully connected bias label
const std::string bias_key = "bias";

teq::Shape gen_rshape (std::vector<teq::DimT> runcoms,
	teq::Shape left, eigen::PairVecT<teq::RankT> lrdims);

template <typename T>
eteq::LinkptrT<T> drop_out (eteq::LinkptrT<T> input, T prob)
{
	auto p = eteq::make_constant_like<T>(prob, input->shape());
	return input * (tenncor::random::rand_binom_one(p) / p);
}

template <typename T, typename ...ARGS>
using LayerBuilderF = std::function<eteq::LayerptrT<T>(eteq::LinkptrT<T>,ARGS...)>;

template <typename T>
using UnaryF = std::function<eteq::LinkptrT<T>(eteq::LinkptrT<T>)>;

template <typename T>
eteq::LayerptrT<T> dense (eteq::LinkptrT<T> input,
	teq::TensptrT weight, teq::TensptrT bias,
	eigen::PairVecT<teq::RankT> dims = {{0, 1}})
{
	auto output = tenncor::nn::fully_connect({input},
		{eteq::to_link<T>(weight)}, eteq::to_link<T>(bias), dims);
	return std::make_shared<eteq::Layer<T>>(teq::Opcode{"_DENSE_LAYER", 0},
		input, output, teq::TensptrsT{weight, bias});
}

template <typename T>
LayerBuilderF<T,eigen::PairVecT<teq::RankT>>
dense_builder (teq::Shape wshape, teq::Shape bshape,
	layr::InitF<T> weight_init, layr::InitF<T> bias_init)
{
	teq::TensptrT weight = weight_init(wshape, weight_key);
	teq::TensptrT bias;
	if (bias_init)
	{
		bias = bias_init(bshape, bias_key);
	}
	return [weight, bias](
			eteq::LinkptrT<T> input, eigen::PairVecT<teq::RankT> dims)
		{
			return dense(input, weight, bias, dims);
		};
}

template <typename T>
eteq::LayerptrT<T> dense (
	eteq::LinkptrT<T> input, std::vector<teq::DimT> hidden_dims,
	layr::InitF<T> weight_init, layr::InitF<T> bias_init,
	eigen::PairVecT<teq::RankT> dims = {{0, 1}})
{
	teq::TensptrT weight = weight_init(gen_rshape(
		hidden_dims, input->shape(), dims), weight_key);
	teq::TensptrT bias;
	if (bias_init)
	{
		bias = bias_init(teq::Shape(hidden_dims), bias_key);
	}
	return dense<T>(input, weight, bias, dims);
}

template <typename T>
struct RBMBuilder final
{
	LayerBuilderF<T> fwd_;

	LayerBuilderF<T> bwd_;
};

/// Returns forward builder, and assigns backward builder
template <typename T>
RBMBuilder<T> rbm_builder (
	teq::DimT nhidden, teq::DimT nvisible,
	layr::InitF<T> weight_init, layr::InitF<T> bias_init)
{
	teq::TensptrT weight = weight_init(
		teq::Shape({nhidden, nvisible}), weight_key);
	teq::TensptrT hbias;
	teq::TensptrT vbias;
	if (bias_init)
	{
		hbias = bias_init(teq::Shape({nhidden}), bias_key);
		vbias = bias_init(teq::Shape({nvisible}), bias_key);
	}
	return RBMBuilder<T>{
		[weight, hbias](eteq::LinkptrT<T> input)
		{
			return dense(input, weight, hbias, {{0, 1}});
		},
		[weight, vbias](eteq::LinkptrT<T> input)
		{
			return dense(input, tenncor::transpose(
				eteq::to_link<T>(weight))->get_tensor(), vbias, {{0, 1}});
		}
	};
}

template <typename T>
eteq::LayerptrT<T> conv (eteq::LinkptrT<T> input,
	std::pair<teq::DimT,teq::DimT> filter_hw,
	teq::DimT in_ncol, teq::DimT out_ncol,
	std::pair<teq::DimT,teq::DimT> zero_padding = {0, 0})
{
	teq::TensptrT weight = unif_xavier_init<T>(1)(teq::Shape({out_ncol,
		in_ncol, filter_hw.second, filter_hw.first}), weight_key);
	teq::TensptrT bias = zero_init<T>()(teq::Shape({out_ncol}), bias_key);
	auto output = tenncor::nn::conv2d(input,
		eteq::to_link<T>(weight), eteq::to_link<T>(bias), zero_padding);
	return std::make_shared<eteq::Layer<T>>(teq::Opcode{"_CONV_LAYER", 0},
		input, output, teq::TensptrsT{weight, bias});
}

template <typename T>
eteq::LayerptrT<T> rnn (eteq::LinkptrT<T> input, teq::DimT hidden_dim,
	UnaryF<T> activation, layr::InitF<T> weight_init,
	layr::InitF<T> bias_init, teq::RankT seq_dim)
{
	teq::Shape inshape = input->shape();

	auto cell = dense_builder(
		teq::Shape({hidden_dim, (teq::DimT) (hidden_dim + inshape.at(0))}),
		teq::Shape({hidden_dim}), weight_init, bias_init);
	auto init_state = eteq::make_variable<T>(
		teq::Shape({hidden_dim}), "init_state");

	teq::DimT nseq = inshape.at(seq_dim);
	if (nseq == 0)
	{
		logs::fatalf("cannot sequence on ambiguous dimension %d on shape %s",
			seq_dim, inshape.to_string().c_str());
	}
	if (seq_dim == 0)
	{
		logs::fatalf("spliting input across 0th dimension... "
			"dense connection will not match");
	}
	std::vector<teq::DimT> slice_shape(inshape.begin(), inshape.end());
	slice_shape[seq_dim] = 1;
	eteq::LinksT<T> states;
	eteq::LinkptrT<T> state = tenncor::best_extend(
		eteq::to_link<T>(init_state), teq::Shape(slice_shape));
	for (teq::DimT i = 0; i < nseq; ++i)
	{
		auto inslice = tenncor::slice(input, i, 1, seq_dim);
		state = activation(cell(tenncor::concat(inslice, state, 0)));
		states.push_back(state);
	}
	auto output = tenncor::concat(states, seq_dim);
	return std::make_shared<eteq::Layer<T>>(teq::Opcode{"_RNN_LAYER", 0},
		input, output, teq::TensptrsT{cell});
}

template <typename T>
eteq::LayerptrT<T> lstm (eteq::LinkptrT<T> input, teq::DimT hidden_dim,
	UnaryF<T> activation, layr::InitF<T> weight_init,
	layr::InitF<T> bias_init, teq::RankT seq_dim)
{
	teq::Shape inshape = input->shape();

	teq::Shape wshape({hidden_dim, (teq::DimT) (hidden_dim + inshape.at(0))});
	teq::Shape bshape({hidden_dim});
	auto ggate = dense_builder(wshape, bshape, weight_init, bias_init);
	auto forgate = dense_builder(wshape, bshape, weight_init, bias_init);
	auto ingate = dense_builder(wshape, bshape, weight_init, bias_init);
	auto outgate = dense_builder(wshape, bshape, weight_init, bias_init);

	teq::DimT nseq = inshape.at(seq_dim);
	if (nseq == 0)
	{
		logs::fatalf("cannot sequence on ambiguous dimension %d on shape %s",
			seq_dim, inshape.to_string().c_str());
	}
	if (seq_dim == 0)
	{
		logs::fatalf("spliting input across 0th dimension... "
			"dense connection will not match");
	}
	teq::Shape stateshape({hidden_dim});
	auto prevstate = eteq::make_constant_scalar<T>(0, stateshape);
	auto prevhidden = eteq::make_constant_scalar<T>(0, stateshape);
	eteq::LinksT<T> states;
	for (teq::DimT i = 0; i < nseq; ++i)
	{
		auto inslice = tenncor::slice(input, i, 1, seq_dim);
		eteq::LinkptrT<T> xc = tenncor::concat(inslice, prevhidden, 0);

		auto gate = tenncor::tanh(ggate(xc));
		auto input = tenncor::sigmoid(ingate(xc));
		auto forget = tenncor::sigmoid(forgate(xc));
		auto output = tenncor::sigmoid(outgate(xc));
		prevstate = gate * input + prevstate * forget;
		prevhidden = prevstate * output;
		states.push_back(prevhidden);
	}
	auto output = tenncor::concat(states, seq_dim);
	return std::make_shared<eteq::Layer<T>>(teq::Opcode{"_LSTM_LAYER", 0},
		input, output, teq::TensptrsT{ggate, forgate, ingate, outgate});
}

template <typename T>
eteq::LayerptrT<T> gru (eteq::LinkptrT<T> input, teq::DimT hidden_dim,
	UnaryF<T> activation, layr::InitF<T> weight_init,
	layr::InitF<T> bias_init, teq::RankT seq_dim)
{
	teq::Shape inshape = input->shape();

	teq::Shape wshape({hidden_dim, (teq::DimT) (hidden_dim + inshape.at(0))});
	teq::Shape bshape({hidden_dim});
	auto ugate = dense_builder(wshape, bshape, weight_init, bias_init);
	auto rgate = dense_builder(wshape, bshape, weight_init, bias_init);
	auto hgate = dense_builder(wshape, bshape, weight_init, bias_init);

	teq::DimT nseq = inshape.at(seq_dim);
	if (nseq == 0)
	{
		logs::fatalf("cannot sequence on ambiguous dimension %d on shape %s",
			seq_dim, inshape.to_string().c_str());
	}
	if (seq_dim == 0)
	{
		logs::fatalf("spliting input across 0th dimension... "
			"dense connection will not match");
	}
	teq::Shape stateshape({hidden_dim});
	auto state = eteq::make_constant_scalar<T>(0, stateshape);
	eteq::LinksT<T> states;
	for (teq::DimT i = 0; i < nseq; ++i)
	{
		auto inslice = tenncor::slice(input, i, 1, seq_dim);
		eteq::LinkptrT<T> xc = tenncor::concat(inslice, state, 0);
		auto update = tenncor::sigmoid(ugate(xc));
		auto reset = tenncor::sigmoid(rgate(xc));
		auto hidden = tenncor::tanh(hgate(
			tenncor::concat(inslice, reset * state, 0)));
		state = update * state + ((T) 1 - update) * hidden;
		states.push_back(state);
	}
	auto output = tenncor::concat(states, seq_dim);
	return std::make_shared<eteq::Layer<T>>(teq::Opcode{"_GRU_LAYER", 0},
		input, output, teq::TensptrsT{ugate, rgate, hgate});
}

}

#endif // LAYR_API_HPP
