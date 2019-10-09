#include "layr/rbm.hpp"
#include "layr/err_approx.hpp"

#include "rocnnet/trainer/sgd_trainer.hpp"

#ifndef RCN_RBM_TRAINER_HPP
#define RCN_RBM_TRAINER_HPP

namespace trainer
{

using ErrorF = std::function<NodeptrT(NodeptrT,NodeptrT)>;

// Bernoulli RBM "error approximation"
// for each (x, err) in leaves
// momentum_next ~ χ * momentum_cur + η * (1 - χ) / err.shape[0] * err
// x_next = x_curr + next_momentum
//
// where η is the learning rate, and χ is discount_factor
layr::AssignGroupsT bbernoulli_approx (const layr::VarErrsT& leaves,
	PybindT learning_rate, PybindT discount_factor,
	std::string root_label = "");

TrainErrF bernoulli_rbm_train (layr::RBM& model, eteq::iSession& sess,
	NodeptrT visible, PybindT learning_rate, PybindT discount_factor,
	ErrorF err_func = ErrorF());

}

#endif // RCN_RBM_TRAINER_HPP
