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
using UnaryF = std::function<eteq::ETensor<T>(eteq::ETensor<T>)>;

template <typename T>
UnaryF<T> dense_builder (teq::Shape inshape,
	std::vector<teq::DimT> hidden_dims,
	layr::InitF<T> weight_init, layr::InitF<T> bias_init,
	eigen::PairVecT<teq::RankT> dims = {{0, 1}})
{
	teq::TensptrT weight = weight_init(gen_rshape(
		hidden_dims, inshape, dims), weight_key);
	teq::TensptrT bias;
	if (bias_init)
	{
		bias = bias_init(teq::Shape(hidden_dims), bias_key);
	}
	return [weight, bias, dims](eteq::ETensor<T> input)
		{
			return tenncor::layer::dense(input,
				eteq::ETensor<T>(weight), eteq::ETensor<T>(bias), dims);
		};
}

template <typename T>
UnaryF<T> conv_builder (std::pair<teq::DimT,teq::DimT> filter_hw,
	teq::DimT in_ncol, teq::DimT out_ncol,
	std::pair<teq::DimT,teq::DimT> zero_padding = {0, 0})
{
	teq::TensptrT weight = unif_xavier_init<T>(1)(teq::Shape({out_ncol,
		in_ncol, filter_hw.second, filter_hw.first}), weight_key);
	teq::TensptrT bias = zero_init<T>()(teq::Shape({out_ncol}), bias_key);
	return [weight, bias, zero_padding](eteq::ETensor<T> input)
		{
			return tenncor::layer::conv(input, eteq::ETensor<T>(weight),
				eteq::ETensor<T>(bias), zero_padding);
		};
}

template <typename T>
UnaryF<T> rnn_builder (teq::DimT indim, teq::DimT hidden_dim,
	UnaryF<T> activation, layr::InitF<T> weight_init,
	layr::InitF<T> bias_init, teq::RankT seq_dim)
{
	auto cell = dense_builder(
		teq::Shape({hidden_dim, (teq::DimT) (hidden_dim + indim)}),
		teq::Shape({hidden_dim}), weight_init, bias_init);
	auto init_state = eteq::make_variable<T>(
		teq::Shape({hidden_dim}), "init_state");
	return [=](eteq::ETensor<T> input)
	{
		teq::Shape inshape = input->shape();
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
		eteq::ETensorsT<T> states;
		eteq::ETensor<T> state = tenncor::best_extend(
			eteq::ETensor<T>(init_state), teq::Shape(slice_shape));
		for (teq::DimT i = 0; i < nseq; ++i)
		{
			auto inslice = tenncor::slice(input, i, 1, seq_dim);
			state = activation(cell(tenncor::concat(inslice, state, 0)));
			states.push_back(state);
		}
		auto output = tenncor::concat(states, seq_dim);
		return eteq::make_layer(teq::Opcode{"_RNN_LAYER", 0}, {input}, output);
	};
}

template <typename T>
UnaryF<T> lstm_builder (teq::DimT indim, teq::DimT hidden_dim,
	UnaryF<T> activation, layr::InitF<T> weight_init,
	layr::InitF<T> bias_init, teq::RankT seq_dim)
{
	teq::Shape wshape({hidden_dim, (teq::DimT) (hidden_dim + indim)});
	teq::Shape bshape({hidden_dim});
	auto ggate = dense_builder(wshape, bshape, weight_init, bias_init);
	auto forgate = dense_builder(wshape, bshape, weight_init, bias_init);
	auto ingate = dense_builder(wshape, bshape, weight_init, bias_init);
	auto outgate = dense_builder(wshape, bshape, weight_init, bias_init);
	return [=](eteq::ETensor<T> input)
	{
		teq::Shape inshape = input->shape();
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
		eteq::ETensorsT<T> states;
		for (teq::DimT i = 0; i < nseq; ++i)
		{
			auto inslice = tenncor::slice(input, i, 1, seq_dim);
			eteq::ETensor<T> xc = tenncor::concat(inslice, prevhidden, 0);

			auto gate = tenncor::tanh(ggate(xc));
			auto input = tenncor::sigmoid(ingate(xc));
			auto forget = tenncor::sigmoid(forgate(xc));
			auto output = tenncor::sigmoid(outgate(xc));
			prevstate = gate * input + prevstate * forget;
			prevhidden = prevstate * output;
			states.push_back(prevhidden);
		}
		auto output = tenncor::concat(states, seq_dim);
		return eteq::make_layer(teq::Opcode{"_LSTM_LAYER", 0}, input, output);
	};
}

template <typename T>
UnaryF<T> gru_builder (teq::DimT indim, eteq::ETensor<T> input,
	teq::DimT hidden_dim, UnaryF<T> activation, layr::InitF<T> weight_init,
	layr::InitF<T> bias_init, teq::RankT seq_dim)
{
	teq::Shape wshape({hidden_dim, (teq::DimT) (hidden_dim + indim)});
	teq::Shape bshape({hidden_dim});
	auto ugate = dense_builder(wshape, bshape, weight_init, bias_init);
	auto rgate = dense_builder(wshape, bshape, weight_init, bias_init);
	auto hgate = dense_builder(wshape, bshape, weight_init, bias_init);
	return [=](eteq::ETensor<T> input)
	{
		teq::Shape inshape = input->shape();
		teq::DimT nseq = inshape.at(seq_dim);
		if (nseq == 0)
		{
			logs::fatalf("cannot sequence on ambiguous dimension %d "
				"on shape %s", seq_dim, inshape.to_string().c_str());
		}
		if (seq_dim == 0)
		{
			logs::fatalf("spliting input across 0th dimension... "
				"dense connection will not match");
		}
		teq::Shape stateshape({hidden_dim});
		auto state = eteq::make_constant_scalar<T>(0, stateshape);
		eteq::ETensorsT<T> states;
		for (teq::DimT i = 0; i < nseq; ++i)
		{
			auto inslice = tenncor::slice(input, i, 1, seq_dim);
			eteq::ETensor<T> xc = tenncor::concat(inslice, state, 0);
			auto update = tenncor::sigmoid(ugate(xc));
			auto reset = tenncor::sigmoid(rgate(xc));
			auto hidden = tenncor::tanh(hgate(
				tenncor::concat(inslice, reset * state, 0)));
			state = update * state + ((T) 1 - update) * hidden;
			states.push_back(state);
		}
		auto output = tenncor::concat(states, seq_dim);
		return eteq::make_layer(teq::Opcode{"_GRU_LAYER", 0}, input, output);
	};
}

template <typename T>
struct RBMBuilder final
{
	UnaryF<T> fwd_;

	UnaryF<T> bwd_;
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
		hbias = bias_init(teq::Shape({nhidden}), "h" + bias_key);
		vbias = bias_init(teq::Shape({nvisible}), "v" + bias_key);
	}
	return RBMBuilder<T>{
		[weight, hbias](eteq::ETensor<T> input)
		{
			return tenncor::layer::dense(input, eteq::ETensor<T>(weight),
				eteq::ETensor<T>(hbias), {{0, 1}});
		},
		[weight, vbias](eteq::ETensor<T> input)
		{
			return tenncor::layer::dense(input, tenncor::transpose(eteq::ETensor<T>(weight)),
				eteq::ETensor<T>(vbias), {{0, 1}});
		}
	};
}

template <typename T>
eteq::ETensor<T> dense (eteq::ETensor<T> input,
	std::vector<teq::DimT> hidden_dims,
	layr::InitF<T> weight_init, layr::InitF<T> bias_init,
	eigen::PairVecT<teq::RankT> dims = {{0, 1}})
{
	return dense_builder<T>(input->shape(), hidden_dims,
		weight_init, bias_init, dims)(input);
}

template <typename T>
eteq::ETensor<T> conv (eteq::ETensor<T> input,
	std::pair<teq::DimT,teq::DimT> filter_hw,
	teq::DimT in_ncol, teq::DimT out_ncol,
	std::pair<teq::DimT,teq::DimT> zero_padding = {0, 0})
{
	return conv_builder<T>(filter_hw, in_ncol, out_ncol, zero_padding)(input);
}

template <typename T>
eteq::ETensor<T> rnn (eteq::ETensor<T> input, teq::DimT hidden_dim,
	UnaryF<T> activation, layr::InitF<T> weight_init,
	layr::InitF<T> bias_init, teq::RankT seq_dim)
{
	return rnn_builder<T>(input->shape().at(0), hidden_dim, activation,
		weight_init, bias_init, seq_dim)(input);
}

template <typename T>
eteq::ETensor<T> lstm (eteq::ETensor<T> input, teq::DimT hidden_dim,
	UnaryF<T> activation, layr::InitF<T> weight_init,
	layr::InitF<T> bias_init, teq::RankT seq_dim)
{
	return lstm_builder<T>(input->shape().at(0), hidden_dim, activation,
		weight_init, bias_init, seq_dim)(input);
}

template <typename T>
eteq::ETensor<T> gru (eteq::ETensor<T> input, teq::DimT hidden_dim,
	UnaryF<T> activation, layr::InitF<T> weight_init,
	layr::InitF<T> bias_init, teq::RankT seq_dim)
{
	return gru_builder<T>(input->shape().at(0), hidden_dim, activation,
		weight_init, bias_init, seq_dim)(input);
}

}

#endif // LAYR_API_HPP
