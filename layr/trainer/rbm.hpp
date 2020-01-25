#include "layr/trainer/trainer.hpp"

#ifndef TRAINER_RBM_HPP
#define TRAINER_RBM_HPP

namespace trainer
{

template <typename T>
eteq::ETensor<T> sample_v2h (
	const layr::RBMLayer<T>& model, eteq::ETensor<T> vis)
{
	return tenncor::random::rand_binom_one(tenncor::sigmoid(
		model.fwd_.connect(vis)));
}

template <typename T>
eteq::ETensor<T> sample_h2v (
	const layr::RBMLayer<T>& model, eteq::ETensor<T> hid)
{
	return tenncor::random::rand_binom_one(tenncor::sigmoid(
		model.bwd_.connect(hid)));
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

	io.visible_mean_ = tenncor::sigmoid(model.bwd_.connect(chain_it));
	io.hidden_mean_ = tenncor::sigmoid(model.fwd_.connect(io.visible_mean_));

	eteq::VarptrsT<T> fcontent = model.fwd_.get_storage();
	eteq::VarptrsT<T> bcontent = model.bwd_.get_storage();
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
TrainErrF<T> rbm (const layr::RBMLayer<T>& model, teq::iSession& sess,
	eteq::ETensor<T> visible, T learning_rate, T discount_factor,
	layr::ErrorF<T> err_func = layr::ErrorF<T>(), size_t cdk = 1)
{
	CDChainIO<T> chain_io(visible);
	layr::VarMapT<T> varerrs = cd_grad_approx<T>(chain_io, model, cdk);
	auto updates = bbernoulli_approx<T>(varerrs, learning_rate, discount_factor);

	teq::TensptrsT to_track;
	teq::TensSetT update_tens;
	to_track.reserve(updates.size());
	for (auto& update : updates)
	{
		to_track.push_back(update.second);
		update_tens.emplace(update.second.get());
	}
	eteq::ETensor<T> error;
	if (err_func)
	{
		error = err_func(chain_io.visible_, chain_io.visible_mean_);
		to_track.push_back(error);
	}
	sess.track(to_track);

	return [&sess, update_tens, error]
	{
		sess.update_target(update_tens);
		if (nullptr == error)
		{
			return teq::ShapedArr<T>{teq::Shape(),std::vector<T>{-1}};
		}
		sess.update_target({error.get()});
		T* data = (T*) error->data();
		teq::Shape shape = error->shape();
		return teq::ShapedArr<T>{shape,
			std::vector<T>(data, data + shape.n_elems()),
		};
	};
}

}

#endif // TRAINER_RBM_HPP
