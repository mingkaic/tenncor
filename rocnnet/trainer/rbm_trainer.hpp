#include "layr/rbm.hpp"
#include "layr/err_approx.hpp"

#include "rocnnet/trainer/sgd_trainer.hpp"

#ifndef RCN_RBM_TRAINER_HPP
#define RCN_RBM_TRAINER_HPP

namespace trainer
{

template <typename T>
eteq::ETensor<T> sample_v2h (
	const layr::RBMBuilder<T>& conn, eteq::ETensor<T> vis)
{
	return tenncor::random::rand_binom_one(conn.fwd_(vis));
}

template <typename T>
eteq::ETensor<T> sample_h2v (
	const layr::RBMBuilder<T>& conn, eteq::ETensor<T> hid)
{
	return tenncor::random::rand_binom_one(conn.bwd_(hid));
}

template <typename T>
eteq::ETensor<T> gibbs_hvh (
	const layr::RBMBuilder<T>& conn, eteq::ETensor<T> hid)
{
	return sample_v2h(conn, sample_h2v(conn, hid));
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
layr::AssignGroupsT<T> bbernoulli_approx (const layr::VarErrsT<T>& assocs,
	T learning_rate, T discount_factor)
{
	// assign momentums before leaves
	size_t nassocs = assocs.size();
	layr::AssignsT<T> assigns;
	assigns.reserve(nassocs * 2);
	for (size_t i = 0; i < nassocs; ++i)
	{
		auto err = assocs[i].second;
		auto momentum = eteq::make_variable_like<T>(0, err, "momentum");

		auto slist = teq::narrow_shape(err->shape());
		teq::DimT shape_factor = slist.empty() ? 1 : slist.back();
		auto momentum_next = discount_factor * eteq::ETensor<T>(momentum) +
			(learning_rate * (1 - discount_factor) / shape_factor) * err;
		auto leaf_next = eteq::ETensor<T>(assocs[i].first) + momentum_next;
		assigns.push_back(layr::VarAssign{momentum, momentum_next});
		assigns.push_back(layr::VarAssign{assocs[i].first, leaf_next});
	}
	return {assigns};
}

template <typename T>
struct CDChainIO final
{
	CDChainIO (eteq::ETensor<T> visible) : visible_(visible) {}

	CDChainIO (eteq::ETensor<T> visible, eteq::ETensor<T> hidden) :
		visible_(visible), hidden_(hidden) {}

	eteq::ETensor<T> visible_;

	eteq::ETensor<T> hidden_ = nullptr;

	eteq::ETensor<T> visible_mean_ = nullptr;

	eteq::ETensor<T> hidden_mean_ = nullptr;
};

/// Contrastive divergence error approximation instead of
/// using backprop calculated gradient
template <typename T>
layr::VarErrsT cd_grad_approx (CDChainIO<T>& io,
	const layr::RBMBuilder<T>& connecter, size_t cdk = 1,
	eteq::VarptrT<T> persistent = nullptr)
{
	if (nullptr == io.visible_)
	{
		logs::fatal("cannot call cd_grad_approx with null visible");
	}
	if (nullptr == io.hidden_)
	{
		io.hidden_ = sample_v2h(connecter, io.visible_);
	}
	auto chain_it = nullptr == persistent ?
		io.hidden_ : eteq::ETensor<T>(persistent);
	for (size_t i = 0; i < cdk - 1; ++i)
	{
		chain_it = gibbs_hvh(connecter, chain_it);
	}

	io.visible_mean_ = connecter.bwd_(chain_it);
	io.hidden_mean_ = connecter.fwd_(io.visible_mean_);

	auto bwd_model = static_cast<eteq::Layer<T>*>(io.visible_mean_);
	auto fwd_model = static_cast<eteq::Layer<T>*>(io.hidden_mean_);
	auto bcontent = bwd_model->get_storage();
	auto fcontent = fwd_model->get_storage();
	std::unordered_map<std::string,eteq::VarptrT<T>> vars;
	for (auto tens : bcontent)
	{
		vars.emplace(tens->to_string(),
			std::static_pointer_cast<eteq::Variable<T>>(tens));
	}
	for (auto tens : fcontent)
	{
		vars.emplace(tens->to_string(),
			std::static_pointer_cast<eteq::Variable<T>>(tens));
	}

	auto grad_w =
		tenncor::matmul(tenncor::transpose(io.visible_), io.hidden_) -
		tenncor::matmul(tenncor::transpose(io.visible_mean_), io.hidden_mean_);
	layr::VarErrsT varerrs = {
		{vars[layr::weight_key], grad_w},
	};

	std::string hid_key = "h" + layr::bias_key;
	std::string vis_key = "v" + layr::bias_key;
	if (estd::has(vars, hid_key))
	{
		auto grad_hb = tenncor::reduce_mean_1d(io.hidden_ - io.hidden_mean_, 1);
		varerrs.push_back({vars[hid_key], grad_hb});
	}
	if (estd::has(vars, vis_key))
	{
		auto grad_vb = tenncor::reduce_mean_1d(io.visible_ - io.visible_mean_, 1);
		varerrs.push_back({vars[vis_key], grad_vb});
	}
	if (nullptr != persistent)
	{
		varerrs.push_back({persistent, gibbs_hvh(model, chain_it)});
	}
	return varerrs;
}

template <typename T>
TrainErrF<T> rbm_train (const layr::RBMBuilder<T>& connecter,
	teq::iSession& sess, eteq::ETensor<T> visible,
	T learning_rate, T discount_factor,
	layr::ErrorF<T> err_func = layr::ErrorF<T>(),
	size_t cdk = 1)
{
	CDChainIO<T> chain_io(visible);
	layr::VarErrsT<T> varerrs = cd_grad_approx<T>(chain_io, model, cdk);
	auto updates = bbernoulli_approx<T>(varerrs, learning_rate, discount_factor);

	teq::TensptrsT to_track;
	to_track.reserve(updates.size() + 1);
	eteq::ETensor<T> error = nullptr;
	if (err_func)
	{
		error = err_func(chain_io.visible_, chain_io.visible_mean_);
		to_track.push_back(error);
	}

	for (auto& assigns : updates)
	{
		for (auto& assign : assigns)
		{
			to_track.push_back(assign.source_);
		}
	}
	sess.track(to_track);

	return [&sess, updates, error]
	{
		assign_groups_preupdate<T>(updates,
			[&](teq::TensSetT& sources)
			{
				sess.update_target(sources);
			});
		if (nullptr == error)
		{
			return teq::ShapedArr<T>{teq::Shape(),std::vector<T>{-1}};
		}
		sess.update_target({error.get()});
		T* data = error->data();
		teq::Shape shape = error->shape();
		return teq::ShapedArr<T>{shape,
			std::vector<T>(data, data + shape.n_elems()),
		};
	};
}

}

#endif // RCN_RBM_TRAINER_HPP
