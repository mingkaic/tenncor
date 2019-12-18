#include "eteq/derive.hpp"

#include "layr/seqmodel.hpp"
#include "layr/err_approx.hpp"

#ifndef RCN_SGD_TRAINER_HPP
#define RCN_SGD_TRAINER_HPP

namespace trainer
{

using NodeUnarF = std::function<eteq::ETensor<PybindT>(eteq::ETensor<PybindT>)>;

using TrainErrF = std::function<teq::ShapedArr<PybindT>(void)>;

TrainErrF sgd_train (layr::iLayer& model, teq::iSession& sess,
	Tensor train_in, Tensor expected_out, layr::ApproxF update,
	layr::ErrorF errfunc = layr::sqr_diff,
	NodeUnarF gradprocess = [](eteq::ETensor<PybindT> in){ return in; });

}

#endif // RCN_SGD_TRAINER_HPP
