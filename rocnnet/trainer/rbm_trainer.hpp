#include "layr/rbm.hpp"
#include "layr/err_approx.hpp"

#include "rocnnet/trainer/sgd_trainer.hpp"

#ifndef RCN_RBM_TRAINER_HPP
#define RCN_RBM_TRAINER_HPP

namespace trainer
{

using ErrorF = std::function<LinkptrT(LinkptrT,LinkptrT)>;

LinkptrT sample_v2h (const layr::RBM& model, LinkptrT v);

LinkptrT sample_h2v (const layr::RBM& model, LinkptrT h);

LinkptrT gibbs_hvh (const layr::RBM& model, LinkptrT h);

// source for below algorithms:
// https://github.com/meownoid/tensorfow-rbm/blob/master/tfrbm/bbrbm.py

// Bernoulli RBM "error approximation"
// for each (x, err) in leaves
// momentum_next ~ χ * momentum_cur + η * (1 - χ) / err.shape[0] * err
// x_next = x_curr + next_momentum
//
// where η is the learning rate, and χ is discount_factor
layr::AssignGroupsT bbernoulli_approx (const layr::VarErrsT& leaves,
	PybindT learning_rate, PybindT discount_factor,
	std::string root_label = "");

struct CDChainIO
{
	CDChainIO (LinkptrT visible) : visible_(visible) {}

	CDChainIO (LinkptrT visible, LinkptrT hidden) :
		visible_(visible), hidden_(hidden) {}

	LinkptrT visible_;

	LinkptrT hidden_ = nullptr;

	LinkptrT visible_mean_ = nullptr;

	LinkptrT hidden_mean_ = nullptr;
};

layr::VarErrsT cd_grad_approx (CDChainIO& io,const layr::RBM& model,
	size_t cdk = 1, eteq::VarptrT<PybindT> persistent = nullptr);

TrainErrF rbm_train (layr::RBM& model, teq::iSession& sess,
	LinkptrT visible, PybindT learning_rate, PybindT discount_factor,
	ErrorF err_func = ErrorF(), size_t cdk = 1);

}

#endif // RCN_RBM_TRAINER_HPP
