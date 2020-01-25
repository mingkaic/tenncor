#include "layr/trainer/trainer.hpp"

#ifndef TRAINER_SGD_HPP
#define TRAINER_SGD_HPP

namespace trainer
{

template <typename T>
eteq::ETensor<T> sgd (const eteq::ELayer<T>& model, eteq::ETensor<T> train_in,
	eteq::ETensor<T> expect_out, layr::ApproxF<T> update,
	layr::ErrorF<T> err_func = layr::sqr_diff<T>,
	layr::UnaryF<T> proc_grad = layr::UnaryF<T>())
{
	eteq::ETensor<T> train_out = model.connect(train_in);
	auto error = err_func(expect_out, train_out);

	eteq::VarptrsT<T> contents = model.get_storage();
	layr::VarMapT<T> vars;
	for (auto var : contents)
	{
		auto derivative = eteq::derive(error, eteq::ETensor<T>(var));
		if (proc_grad)
		{
			derivative = proc_grad(derivative);
		}
		vars.emplace(var, derivative);
	}
	auto updates = update(vars);
	teq::TensMapT<teq::TensptrT> umap;
	eteq::ETensorsT<T> deps;
	deps.reserve(updates.size());
	for (auto& update : updates)
	{
		umap.emplace(update.first.get(), update.second);
		deps.push_back(update.second);
	}
	// depend on assigns for variables not trailed in error
	return tenncor::depends(eteq::trail(error, umap), deps);
}

}

#endif // TRAINER_SGD_HPP
