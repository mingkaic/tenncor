#include "eteq/grader.hpp"

#include "layr/seqmodel.hpp"
#include "layr/err_approx.hpp"

#ifndef RCN_SGD_TRAINER_HPP
#define RCN_SGD_TRAINER_HPP

namespace trainer
{

using NodeUnarF = std::function<eteq::LinkptrT<PybindT>(eteq::LinkptrT<PybindT>)>;

using TrainErrF = std::function<teq::ShapedArr<PybindT>(void)>;

TrainErrF sgd_train (layr::iLayer& model, teq::iSession& sess,
	LinkptrT train_in, LinkptrT expected_out, layr::ApproxF update,
	layr::ErrorF errfunc = layr::sqr_diff,
	NodeUnarF gradprocess = [](eteq::LinkptrT<PybindT> in){ return in; });

}

#endif // RCN_SGD_TRAINER_HPP
