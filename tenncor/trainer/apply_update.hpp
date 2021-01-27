
#ifndef TRAINER_SGD_HPP
#define TRAINER_SGD_HPP

#include "tenncor/tenncor.hpp"

namespace trainer
{

/// Return node that needs to be calculated for every training step
/// The node value contains the error computed from the err_func input
template <typename T>
eteq::ETensor apply_update (const eteq::ETensorsT& models,
	layr::ApproxF<T> update, layr::ErrorF<T> err_func,
	const global::CfgMapptrT& ctx = global::context())
{
	auto error = err_func(models);
	eteq::EVariablesT vars;
	for (auto& model : models)
	{
		auto temp_vars = layr::get_storage(model);
		std::transform(temp_vars.begin(), temp_vars.end(),
			std::back_inserter(vars),
			[&](eteq::VarptrT var)
			{
				return eteq::EVariable(var, model.get_context());
			});
	}
	auto updates = update(error,
		eteq::EVariablesT(vars.begin(), vars.end()));
	teq::OwnMapT umap;
	eteq::ETensorsT deps;
	deps.reserve(updates.size());
	for (auto& update : updates)
	{
		umap.emplace(update.first.get(), update.second);
		deps.push_back(update.second);
	}
	// depend on assigns for variables not trailed in error
	return TenncorAPI(ctx).identity(layr::trail(error, umap), deps);
}

}

#endif // TRAINER_SGD_HPP
