#include "layr/trainer/trainer.hpp"

#ifndef TRAINER_SGD_HPP
#define TRAINER_SGD_HPP

namespace trainer
{

template <typename T>
TrainErrF<T> sgd (const eteq::ELayer<T>& model, teq::iSession& sess,
	eteq::ETensor<T> train_in, eteq::ETensor<T> expect_out,
	layr::ApproxF<T> update, layr::ErrorF<T> err_func = layr::sqr_diff<T>,
	layr::UnaryF<T> proc_grad = layr::UnaryF<T>())
{
	eteq::ETensor<T> train_out = model.connect(train_in);
	auto error = err_func(expect_out, train_out);

	eteq::VarptrsT<T> contents = model.get_storage();
	layr::VarErrsT<T> vars;
	for (auto var : contents)
	{
		auto derivative = eteq::derive(error, eteq::ETensor<T>(var));
		if (proc_grad)
		{
			derivative = proc_grad(derivative);
		}
		vars.push_back({var, derivative});
	}
	auto updates = update(vars);

	teq::TensptrsT track_batch(updates.begin(), updates.end());
	track_batch.push_back(train_out);
	track_batch.push_back(error);
	sess.track(track_batch);

	teq::TensSetT update_tens;
	std::transform(updates.begin(), updates.end(),
		std::inserter(update_tens, update_tens.end()),
		[](eteq::ETensor<T> etens)
		{
			return etens.get();
		});

	return [&sess, update_tens, error](void)
	{
		sess.update_target(update_tens);
		sess.update_target({error.get()});
		T* data = (T*) error->data();
		teq::Shape shape = error->shape();
		return teq::ShapedArr<T>{shape,
			std::vector<T>(data, data + shape.n_elems()),
		};
	};
}

}

#endif // TRAINER_SGD_HPP
