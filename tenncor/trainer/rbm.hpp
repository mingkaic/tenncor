
#ifndef TRAINER_RBM_HPP
#define TRAINER_RBM_HPP

#include "tenncor/tenncor.hpp"

namespace trainer
{

template <typename T>
eteq::ETensor sample_v2h (
	const layr::RBMLayer<T>& model, eteq::ETensor vis)
{
	return tenncor().random.rand_binom_one(
		tenncor().sigmoid(model.connect(vis)));
}

template <typename T>
eteq::ETensor sample_h2v (
	const layr::RBMLayer<T>& model, eteq::ETensor hid)
{
	return tenncor().random.rand_binom_one(
		tenncor().sigmoid(model.backward_connect(hid)));
}

template <typename T>
eteq::ETensor gibbs_hvh (
	const layr::RBMLayer<T>& model, eteq::ETensor hid)
{
	return sample_v2h(model, sample_h2v(model, hid));
}

// source for below algorithms:
// https://github.com/meownoid/tensorfow-rbm/blob/master/tfrbm/bbrbm.py

// Bernoulli RBM "error approximation"
// for each (x, err) in leaves
// momentum_next ~ χ * momentum_cur + η * (1 - χ) / err.shape[0] * err
// x_next = x_curr + next_momentum
//
// where η is the learning rate, and χ is discount_factor
template <typename T>
layr::VarErrsT<T> bbernoulli_approx (const layr::VarErrsT<T>& assocs,
	T learning_rate, T discount_factor,
	global::CfgMapptrT context = global::context())
{
	// assign momentums before leaves
	layr::VarErrsT<T> assigns;
	for (const auto& verrs : assocs)
	{
		auto err = verrs.second;
		auto slist = teq::narrow_shape(err->shape());
		teq::DimT shape_factor = slist.empty() ? 1 : slist.back();

		auto momentum = eteq::make_variable_like<T>(0, err, "momentum", context);
		auto momentum_next = discount_factor * momentum +
			(learning_rate * (1 - discount_factor) / shape_factor) * err;

		assigns.push_back({verrs.first,
			tenncor().assign_add(eteq::EVariable<T>(verrs.first),
				tenncor().assign(momentum, momentum_next))});
	}
	return assigns;
}

template <typename T>
struct CDChainIO final
{
	CDChainIO (eteq::ETensor visible) : visible_(visible) {}

	CDChainIO (eteq::ETensor visible, eteq::ETensor hidden) :
		visible_(visible), hidden_(hidden) {}

	eteq::ETensor visible_;

	eteq::ETensor hidden_;

	eteq::ETensor visible_mean_;

	eteq::ETensor hidden_mean_;
};

/// Contrastive divergence error approximation instead of
/// using backprop calculated gradient
template <typename T>
layr::VarErrsT<T> cd_grad_approx (CDChainIO<T>& io,
	const layr::RBMLayer<T>& model, size_t cdk = 1,
	eteq::VarptrT<T> persistent = nullptr,
	global::CfgMapptrT context = global::context())
{
	if (nullptr == io.visible_)
	{
		global::fatal("cannot call cd_grad_approx with null visible");
	}
	if (nullptr == io.hidden_)
	{
		io.hidden_ = sample_v2h(model, io.visible_);
	}
	auto chain_it = nullptr == persistent ?
		io.hidden_ : eteq::ETensor(persistent, context);
	for (size_t i = 0; i < cdk - 1; ++i)
	{
		chain_it = gibbs_hvh(model, chain_it);
	}

	io.visible_mean_ = tenncor().sigmoid(model.backward_connect(chain_it));
	io.hidden_mean_ = tenncor().sigmoid(model.connect(io.visible_mean_));

	eteq::VarptrsT<T> fcontent = layr::get_storage<T>(model.fwd_);
	eteq::VarptrsT<T> bcontent = layr::get_storage<T>(model.bwd_);
	types::StrUMapT<eteq::VarptrT<T>> vars;
	for (eteq::VarptrT<T> var : fcontent)
	{
		vars.emplace(var->to_string(), var);
	}
	for (eteq::VarptrT<T> var : bcontent)
	{
		vars.emplace(var->to_string(), var);
	}

	auto grad_w =
		tenncor().matmul(tenncor().transpose(io.visible_), io.hidden_) -
		tenncor().matmul(tenncor().transpose(io.visible_mean_), io.hidden_mean_);
	layr::VarErrsT<T> varerrs = {
		{eteq::EVariable<T>(vars[layr::weight_label], context), grad_w},
	};

	std::string hid_key = "h" + layr::bias_label;
	std::string vis_key = "v" + layr::bias_label;
	if (estd::has(vars, hid_key))
	{
		auto grad_hb = tenncor().reduce_mean_1d(io.hidden_ - io.hidden_mean_, 1);
		varerrs.push_back({eteq::EVariable<T>(vars[hid_key], context), grad_hb});
	}
	if (estd::has(vars, vis_key))
	{
		auto grad_vb = tenncor().reduce_mean_1d(io.visible_ - io.visible_mean_, 1);
		varerrs.push_back({eteq::EVariable<T>(vars[vis_key], context), grad_vb});
	}
	if (nullptr != persistent)
	{
		varerrs.push_back({eteq::EVariable<T>(persistent, context), gibbs_hvh(model, chain_it)});
	}
	return varerrs;
}

template <typename T>
eteq::ETensor rbm (const layr::RBMLayer<T>& model,
	eteq::ETensor visible, T learning_rate, T discount_factor,
	layr::BErrorF err_func = [](const eteq::ETensor& a, const eteq::ETensor& b)
	{
		return tenncor().error.sqr_diff(a, b);
	}, size_t cdk = 1,
	global::CfgMapptrT context = global::context())
{
	CDChainIO<T> chain_io(visible);
	layr::VarErrsT<T> varerrs = cd_grad_approx<T>(chain_io, model, cdk, nullptr, context);
	auto updates = bbernoulli_approx<T>(varerrs, learning_rate, discount_factor, context);
	teq::OwnMapT umap;
	for (auto& update : updates)
	{
		umap.emplace(update.first.get(), update.second);
	}
	eteq::ETensor error = err_func(chain_io.visible_, chain_io.visible_mean_);
	return layr::trail(error, umap);
}

}

#endif // TRAINER_RBM_HPP
