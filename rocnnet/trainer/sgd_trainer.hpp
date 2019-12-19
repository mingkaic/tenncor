#include "eteq/derive.hpp"

#include "layr/api.hpp"
#include "layr/err_approx.hpp"

#ifndef RCN_SGD_TRAINER_HPP
#define RCN_SGD_TRAINER_HPP

namespace trainer
{

template <typename T>
using TrainErrF = std::function<teq::ShapedArr<T>(void)>;

template <typename T=PybindT>
TrainErrF<T> sgd_train (layr::UnaryF<T> connect, teq::iSession& sess,
	eteq::ETensor<T> train_in, eteq::ETensor<T> expected_out,
	layr::ApproxF<T> update, layr::ErrorF<T> errfunc = layr::sqr_diff<T>,
	layr::UnaryF<T> proc_grad = layr::UnaryF<T>())
{
	eteq::ETensor<T> train_out = connect(train_in);
	auto error = errfunc(expected_out, train_out);

	auto model = static_cast<eteq::Layer<T>*>(train_out.get());
	auto contents = model->get_storage();
	layr::VarErrsT<T> vars;
	for (auto tens : contents)
	{
		auto var = std::static_pointer_cast<eteq::Variable<T>>(tens);
		auto derivative = eteq::derive(error, eteq::ETensor<T>(var));
		if (proc_grad)
		{
			derivative = proc_grad(derivative);
		}
		vars.push_back({var, derivative});
	}
	auto updates = update(vars);

	teq::TensptrsT track_batch = {
		train_out,
		error,
	};
	for (layr::AssignsT<T>& assigns : updates)
	{
		for (layr::VarAssign<T>& assign : assigns)
		{
			track_batch.push_back(assign.source_);
		}
	}
	sess.track(track_batch);

	return [&sess, updates, error](void)
	{
		layr::assign_groups_preupdate<T>(updates,
			[&](teq::TensSetT& sources)
			{
				sess.update_target(sources);
			});
		sess.update_target({error.get()});
		T* data = error->data();
		teq::Shape shape = error->shape();
		return teq::ShapedArr<T>{shape,
			std::vector<T>(data, data + shape.n_elems()),
		};
	};
}

}

#endif // RCN_SGD_TRAINER_HPP
