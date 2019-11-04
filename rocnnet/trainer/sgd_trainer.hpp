#include "eteq/grader.hpp"

#include "layr/seqmodel.hpp"
#include "layr/err_approx.hpp"

#ifndef RCN_SGD_TRAINER_HPP
#define RCN_SGD_TRAINER_HPP

namespace trainer
{

using NodeUnarF = std::function<eteq::NodeptrT<PybindT>(eteq::NodeptrT<PybindT>)>;

using TrainErrF = std::function<teq::ShapedArr<PybindT>(void)>;

TrainErrF sgd_train (layr::SequentialModel& model, teq::iSession& sess,
	NodeptrT train_in, NodeptrT expected_out, layr::ApproxF update,
	layr::ErrorF errfunc = layr::sqr_diff,
	NodeUnarF gradprocess = [](eteq::NodeptrT<PybindT> in){ return in; });

}

#endif // RCN_SGD_TRAINER_HPP
