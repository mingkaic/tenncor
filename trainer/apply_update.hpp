#include "trainer/trainer.hpp"

#ifndef TRAINER_SGD_HPP
#define TRAINER_SGD_HPP

namespace trainer
{

template <typename T>
eteq::ETensor<T> apply_update (const eteq::ETensorsT<T>& models,
	layr::ApproxF<T> update, layr::ErrorF<T> err_func)
{
	auto error = err_func(models);
	eteq::VarptrsT<T> vars;
	for (auto& model : models)
	{
		auto temp_vars = eteq::get_storage(model);
		vars.insert(vars.end(),
			temp_vars.begin(), temp_vars.end());
	}
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
