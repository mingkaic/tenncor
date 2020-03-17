#include "trainer/trainer.hpp"

#ifndef TRAINER_RBM_HPP
#define TRAINER_RBM_HPP

namespace trainer
{

template <typename T>
eteq::ETensor<T> sample_v2h (
	const layr::RBMLayer<T>& model, eteq::ETensor<T> vis)
{
	return tenncor::random::rand_binom_one(
		tenncor::sigmoid(model.connect(vis)));
}

template <typename T>
eteq::ETensor<T> sample_h2v (
	const layr::RBMLayer<T>& model, eteq::ETensor<T> hid)
{
	return tenncor::random::rand_binom_one(
		tenncor::sigmoid(model.backward_connect(hid)));
}

template <typename T>
eteq::ETensor<T> gibbs_hvh (
	const layr::RBMLayer<T>& model, eteq::ETensor<T> hid)
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
layr::VarMapT<T> bbernoulli_approx (const layr::VarMapT<T>& assocs,
	T learning_rate, T discount_factor)
{
	// assign momentums before leaves
	layr::VarMapT<T> assigns;
	for (const auto& verrs : assocs)
	{
		auto err = verrs.second;
		auto slist = teq::narrow_shape(err->shape());
		teq::DimT shape_factor = slist.empty() ? 1 : slist.back();

		auto momentum = eteq::make_variable_like<T>(0, err, "momentum");
		auto momentum_next = discount_factor * momentum +
			(learning_rate * (1 - discount_factor) / shape_factor) * err;

		assigns.emplace(verrs.first,
			tenncor::assign_add(eteq::EVariable<T>(verrs.first),
				tenncor::assign(momentum, momentum_next)));
	}
	return assigns;
}

template <typename T>
struct CDChainIO final
{
	CDChainIO (eteq::ETensor<T> visible) : visible_(visible) {}

	CDChainIO (eteq::ETensor<T> visible, eteq::ETensor<T> hidden) :
		visible_(visible), hidden_(hidden) {}

	eteq::ETensor<T> visible_;

	eteq::ETensor<T> hidden_;

	eteq::ETensor<T> visible_mean_;

	eteq::ETensor<T> hidden_mean_;
};

/// Contrastive divergence error approximation instead of
/// using backprop calculated gradient
template <typename T>
layr::VarMapT<T> cd_grad_approx (CDChainIO<T>& io,
	const layr::RBMLayer<T>& model, size_t cdk = 1,
	eteq::VarptrT<T> persistent = nullptr)
{
	if (nullptr == io.visible_)
	{
		teq::fatal("cannot call cd_grad_approx with null visible");
	}
	if (nullptr == io.hidden_)
	{
		io.hidden_ = sample_v2h(model, io.visible_);
	}
	auto chain_it = nullptr == persistent ?
		io.hidden_ : eteq::ETensor<T>(persistent);
	for (size_t i = 0; i < cdk - 1; ++i)
	{
		chain_it = gibbs_hvh(model, chain_it);
	}

	io.visible_mean_ = tenncor::sigmoid(model.backward_connect(chain_it));
	io.hidden_mean_ = tenncor::sigmoid(model.connect(io.visible_mean_));

	eteq::VarptrsT<T> fcontent = eteq::get_storage(model.fwd_);
	eteq::VarptrsT<T> bcontent = eteq::get_storage(model.bwd_);
	std::unordered_map<std::string,eteq::VarptrT<T>> vars;
	for (eteq::VarptrT<T> var : fcontent)
	{
		vars.emplace(var->to_string(), var);
	}
	for (eteq::VarptrT<T> var : bcontent)
	{
		vars.emplace(var->to_string(), var);
	}

	auto grad_w =
		tenncor::matmul(tenncor::transpose(io.visible_), io.hidden_) -
		tenncor::matmul(tenncor::transpose(io.visible_mean_), io.hidden_mean_);
	layr::VarMapT<T> varerrs = {
		{vars[layr::weight_label], grad_w},
	};

	std::string hid_key = "h" + layr::bias_label;
	std::string vis_key = "v" + layr::bias_label;
	if (estd::has(vars, hid_key))
	{
		auto grad_hb = tenncor::reduce_mean_1d(io.hidden_ - io.hidden_mean_, 1);
		varerrs.emplace(eteq::EVariable<T>(vars[hid_key]), grad_hb);
	}
	if (estd::has(vars, vis_key))
	{
		auto grad_vb = tenncor::reduce_mean_1d(io.visible_ - io.visible_mean_, 1);
		varerrs.emplace(eteq::EVariable<T>(vars[vis_key]), grad_vb);
	}
	if (nullptr != persistent)
	{
		varerrs.emplace(eteq::EVariable<T>(persistent), gibbs_hvh(model, chain_it));
	}
	return varerrs;
}

template <typename T>
eteq::ETensor<T> rbm (const layr::RBMLayer<T>& model,
	eteq::ETensor<T> visible, T learning_rate, T discount_factor,
	layr::ErrorF<T> err_func = tenncor::error::sqr_diff<T>, size_t cdk = 1)
{
	CDChainIO<T> chain_io(visible);
	layr::VarMapT<T> varerrs = cd_grad_approx<T>(chain_io, model, cdk);
	auto updates = bbernoulli_approx<T>(varerrs, learning_rate, discount_factor);
	teq::TensMapT<teq::TensptrT> umap;
	eteq::ETensorsT<T> deps;
	deps.reserve(updates.size());
	for (auto& update : updates)
	{
		umap.emplace(update.first.get(), update.second);
		deps.push_back(update.second);
	}
	eteq::ETensor<T> error = err_func(chain_io.visible_, chain_io.visible_mean_);
	return tenncor::depends(eteq::trail(error, umap), deps);
}

}

#endif // TRAINER_RBM_HPP
