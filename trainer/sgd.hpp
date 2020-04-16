#include "trainer/trainer.hpp"

#ifndef TRAINER_SGD_HPP
#define TRAINER_SGD_HPP

namespace trainer
{

template <typename T>
eteq::ETensor<T> sgd (
	const eteq::ETensor<T>& model, eteq::ETensor<T> train_in,
	eteq::ETensor<T> expect_out, layr::ApproxF<T> update,
	layr::ErrorF<T> err_func = tenncor::error::sqr_diff<T>)
{
	eteq::ETensor<T> train_out = eteq::connect(model, train_in);
	auto error = err_func(expect_out, train_out);

	eteq::VarptrsT<T> vars = eteq::get_storage(model);
	auto updates = update(error,
		eteq::EVariablesT<T>(vars.begin(), vars.end()));
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
