#include "trainer/trainer.hpp"

#ifndef TRAINER_SGD_HPP
#define TRAINER_SGD_HPP

namespace trainer
{

/// Return node that needs to be calculated for every training step
/// The node value contains the error computed from the err_func input
template <typename T>
eteq::ETensor<T> apply_update (const eteq::ETensorsT<T>& models,
	layr::ApproxF<T> update, layr::ErrorF<T> err_func,
	const global::CfgMapptrT& ctx = global::context())
{
	auto error = err_func(models);
	eteq::EVariablesT<T> vars;
	for (auto& model : models)
	{
		auto temp_vars = eteq::get_storage(model);
		std::transform(temp_vars.begin(), temp_vars.end(),
			std::back_inserter(vars),
			[&](eteq::VarptrT<T> var)
			{
				return eteq::EVariable<T>(var, model.get_context());
			});
	}
	auto updates = update(error,
		eteq::EVariablesT<T>(vars.begin(), vars.end()));
	teq::OwnMapT umap;
	eteq::ETensorsT<T> deps;
	deps.reserve(updates.size());
	for (auto& update : updates)
	{
		umap.emplace(update.first.get(), update.second);
		deps.push_back(update.second);
	}
	// depend on assigns for variables not trailed in error
	return TenncorAPI<T>(ctx).depends(eteq::trail(error, umap), deps);
}

}

#endif // TRAINER_SGD_HPP