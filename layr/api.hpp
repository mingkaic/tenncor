///
/// api.hpp
/// layr
///
/// Purpose:
/// Utility APIs for creating layers
///

#include "eteq/generated/api.hpp"
#include "eteq/serialize.hpp"

#include "layr/init.hpp"

#ifndef LAYR_API_HPP
#define LAYR_API_HPP

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
eteq::ELayer<T> dense (teq::Shape inshape, std::vector<teq::DimT> hidden_dims,
	layr::InitF<T> weight_init, layr::InitF<T> bias_init,
	eigen::PairVecT<teq::RankT> dims = {{0, 1}})
{
	eteq::ETensor<T> input(eteq::make_variable_scalar<T>(0, inshape, input_label));
	eteq::EVariable<T> weight = weight_init(gen_rshape(
		hidden_dims, inshape, dims), weight_label);
	eteq::EVariable<T> bias;
	if (bias_init)
	{
		bias = bias_init(teq::Shape(hidden_dims), bias_label);
	}
	auto output = tenncor::layer::dense(input, weight, bias, dims);
	auto f = std::static_pointer_cast<teq::iFunctor>((teq::TensptrT) output);
	return eteq::ELayer<T>(f, input);
}

template <typename T>
eteq::ELayer<T> conv (std::pair<teq::DimT,teq::DimT> filter_hw,
	teq::DimT in_ncol, teq::DimT out_ncol,
	std::pair<teq::DimT,teq::DimT> zero_padding = {0, 0})
{
	// image must be in form [in, iwidth, iheight, batch]
	eteq::ETensor<T> input(eteq::make_variable_scalar<T>(0,
		teq::Shape({in_ncol, filter_hw.second, filter_hw.first, 1}), input_label));
	eteq::EVariable<T> weight = unif_xavier_init<T>(1)(teq::Shape({out_ncol,
		in_ncol, filter_hw.second, filter_hw.first}), weight_label);
	eteq::EVariable<T> bias = zero_init<T>()(teq::Shape({out_ncol}), bias_label);
	auto output = tenncor::layer::conv(input, weight, bias, zero_padding);
	auto f = std::static_pointer_cast<teq::iFunctor>((teq::TensptrT) output);
	return eteq::ELayer<T>(f, input);
}

template <typename T>
eteq::ELayer<T> rnn (teq::DimT indim, teq::DimT hidden_dim,
	UnaryF<T> activation, teq::DimT nseq, layr::InitF<T> weight_init,
	layr::InitF<T> bias_init, teq::RankT seq_dim)
{
	// input needs to specify number of sequences,
	// since graph topography can't be traced
	std::vector<teq::DimT> inslist(teq::rank_cap, 1);
	inslist[0] = indim;
	inslist[seq_dim] = nseq;
	eteq::ETensor<T> input(eteq::make_variable_scalar<T>(
		0, teq::Shape(inslist), input_label));

	auto cell = dense<T>(teq::Shape({(teq::DimT) (hidden_dim + indim)}),
		{hidden_dim}, weight_init, bias_init);
	teq::TensptrT croot = activation(eteq::ETensor<T>(cell.root()));
	cell = eteq::ELayer<T>(std::static_pointer_cast<
		teq::iFunctor>(croot), cell.input());

	auto init_state = eteq::make_variable<T>(
		teq::Shape({hidden_dim}), "init_state");
	eteq::ETensor<T> state = tenncor::extend_like(init_state,
		tenncor::slice(input, 0, 1, seq_dim));

	auto output = tenncor::layer::rnn(input, state, cell, seq_dim);
	auto f = std::static_pointer_cast<teq::iFunctor>((teq::TensptrT) output);
	return eteq::ELayer<T>(f, input);
}

template <typename T>
eteq::ELayer<T> lstm (teq::DimT indim, teq::DimT hidden_dim,
	teq::DimT nseq, layr::InitF<T> weight_init,
	layr::InitF<T> bias_init, teq::RankT seq_dim)
{
	// input needs to specify number of sequences,
	// since graph topography can't be traced
	std::vector<teq::DimT> inslist(teq::rank_cap, 1);
	inslist[0] = indim;
	inslist[seq_dim] = nseq;
	eteq::ETensor<T> input(eteq::make_variable_scalar<T>(
		0, teq::Shape(inslist), input_label));

	teq::Shape inshape({(teq::DimT) (hidden_dim + indim)});
	std::vector<teq::DimT> hid_dims = {hidden_dim};
	auto ggate = dense<T>(inshape, hid_dims, weight_init, bias_init);
	auto forgate = dense<T>(inshape, hid_dims, weight_init, bias_init);
	auto ingate = dense<T>(inshape, hid_dims, weight_init, bias_init);
	auto outgate = dense<T>(inshape, hid_dims, weight_init, bias_init);

	teq::Shape stateshape({hidden_dim});
	auto state = eteq::make_constant_scalar<T>(0, stateshape);
	auto hidden = eteq::make_constant_scalar<T>(0, stateshape);

	auto output = tenncor::layer::lstm(input, state, hidden,
		ggate, forgate, ingate, outgate, seq_dim);
	auto f = std::static_pointer_cast<teq::iFunctor>((teq::TensptrT) output);
	return eteq::ELayer<T>(f, input);
}

template <typename T>
eteq::ELayer<T> gru (teq::DimT indim, teq::DimT hidden_dim,
	teq::DimT nseq, layr::InitF<T> weight_init,
	layr::InitF<T> bias_init, teq::RankT seq_dim)
{
	// input needs to specify number of sequences,
	// since graph topography can't be traced
	std::vector<teq::DimT> inslist(teq::rank_cap, 1);
	inslist[0] = indim;
	inslist[seq_dim] = nseq;
	eteq::ETensor<T> input(eteq::make_variable_scalar<T>(
		0, teq::Shape(inslist), input_label));

	teq::Shape inshape({(teq::DimT) (hidden_dim + indim)});
	std::vector<teq::DimT> hid_dims = {hidden_dim};
	auto ugate = dense<T>(inshape, hid_dims, weight_init, bias_init);
	auto rgate = dense<T>(inshape, hid_dims, weight_init, bias_init);
	auto hgate = dense<T>(inshape, hid_dims, weight_init, bias_init);

	auto state = eteq::make_constant_scalar<T>(0, teq::Shape({hidden_dim}));

	auto output = tenncor::layer::gru(input, state,
		ugate, rgate, hgate, seq_dim);
	auto f = std::static_pointer_cast<teq::iFunctor>((teq::TensptrT) output);
	return eteq::ELayer<T>(f, input);
}

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

/// Returns forward builder, and assigns backward builder
template <typename T>
RBMLayer<T> rbm (teq::DimT nvisible, teq::DimT nhidden,
	layr::InitF<T> weight_init, layr::InitF<T> bias_init)
{
	eteq::ETensor<T> fwdinput(eteq::make_variable_scalar<T>(
		0, teq::Shape({nvisible}), input_label));
	eteq::ETensor<T> bwdinput(eteq::make_variable_scalar<T>(
		0, teq::Shape({nhidden}), input_label));
	eteq::EVariable<T> weight = weight_init(
		teq::Shape({nhidden, nvisible}), weight_label);
	eteq::EVariable<T> hbias;
	eteq::EVariable<T> vbias;
	if (bias_init)
	{
		hbias = bias_init(teq::Shape({nhidden}), "h" + bias_label);
		vbias = bias_init(teq::Shape({nvisible}), "v" + bias_label);
	}
	auto fwd = tenncor::layer::dense(fwdinput, weight, hbias, {{0, 1}});
	auto bwd = tenncor::layer::dense(
		bwdinput, tenncor::transpose(weight), vbias, {{0, 1}});
	auto ffwd = std::static_pointer_cast<teq::iFunctor>((teq::TensptrT) fwd);
	auto fbwd = std::static_pointer_cast<teq::iFunctor>((teq::TensptrT) bwd);
	return RBMLayer<T>{eteq::ELayer<T>(ffwd, fwdinput), eteq::ELayer<T>(fbwd, bwdinput)};
}

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
		logs::fatal("cannot link no layers");
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
		logs::fatal("cannot link no layers");
	}
	eteq::ETensor<T> input = layers.front().input();
	return link<T>(layers, input);
}

}

#endif // LAYR_API_HPP
