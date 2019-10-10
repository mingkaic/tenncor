#include "layr/rbm.hpp"
#include "layr/err_approx.hpp"

#include "rocnnet/trainer/sgd_trainer.hpp"

#ifndef RCN_RBM_TRAINER_HPP
#define RCN_RBM_TRAINER_HPP

namespace trainer
{

using ErrorF = std::function<NodeptrT(NodeptrT,NodeptrT)>;

NodeptrT sample_v2h (const layr::RBM& model, NodeptrT v);

NodeptrT sample_h2v (const layr::RBM& model, NodeptrT h);

NodeptrT gibbs_hvh (const layr::RBM& model, NodeptrT h);

layr::VarErrsT cd_grad_approx (layr::RBM& model, NodeptrT visible,
	size_t cdk = 1, eteq::VarptrT<PybindT> persistent = nullptr);

// Bernoulli RBM "error approximation"
// for each (x, err) in leaves
// momentum_next ~ χ * momentum_cur + η * (1 - χ) / err.shape[0] * err
// x_next = x_curr + next_momentum
//
// where η is the learning rate, and χ is discount_factor
layr::AssignGroupsT bbernoulli_approx (const layr::VarErrsT& leaves,
	PybindT learning_rate, PybindT discount_factor,
	std::string root_label = "");

// source: https://github.com/meownoid/tensorfow-rbm/blob/master/tfrbm/bbrbm.py
TrainErrF bernoulli_rbm_train (layr::RBM& model, eteq::iSession& sess,
	NodeptrT visible, PybindT learning_rate, PybindT discount_factor,
	ErrorF err_func = ErrorF(), size_t cdk = 1);

}

#endif // RCN_RBM_TRAINER_HPP
