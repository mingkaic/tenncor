#include "tenncor/VERSION.hpp"
#include "tenncor/serial.hpp"

#ifdef TENNCOR_SERIAL_HPP

namespace tcr
{

inline const std::string app_name = "tenncor";
inline const std::string tenncor_dom = "com.mingkaic.tenncor";

distr::ox::TopographyT save_model (
	onnx::ModelProto& pb_model,
	const eteq::ETensorsT& roots,
	const onnx::TensptrIdT& identified)
{
	pb_model.set_ir_version(onnx::IR_VERSION);
	pb_model.set_producer_name(app_name);
	pb_model.set_producer_version(TENNCOR_VERSION);
	pb_model.set_domain(tenncor_dom);
	pb_model.set_model_version(onnx::IR_VERSION);
	// onnx::OperatorSetIdProto* opset = pb_model.add_opset_import();
	// opset->set_domain(tenncor_dom);
	// opset->set_version(onnx::IR_VERSION);
	if (roots.empty())
	{
		return distr::ox::TopographyT{};
	}
	const global::CfgMapptrT& ctx = roots.front().get_context();
	teq::TensptrsT rootens(roots.begin(), roots.end());
	if (auto mgr = get_distrmgr(ctx))
	{
		return distr::get_oxsvc(*mgr).save_graph(
			*pb_model.mutable_graph(), rootens, identified);
	}
	onnx::TensIdT identified_raw;
	for (auto& id : identified)
	{
		identified_raw.insert({id.left.get(), id.right});
	}
	serial::save_graph(*pb_model.mutable_graph(), rootens, identified_raw);
	return distr::ox::TopographyT{};
}

eteq::ETensorsT load_model (
	onnx::TensptrIdT& identified_tens,
	const onnx::ModelProto& pb_model,
	const global::CfgMapptrT& ctx,
	const distr::ox::TopographyT& topography)
{
	teq::TensptrsT tens;
	if (auto mgr = get_distrmgr(ctx))
	{
		tens = distr::get_oxsvc(*mgr).load_graph(
			identified_tens, pb_model.graph(), topography);
	}
	else
	{
		tens = serial::load_graph(identified_tens, pb_model.graph());
	}
	eteq::ETensorsT etens;
	etens.reserve(tens.size());
	std::transform(tens.begin(), tens.end(), std::back_inserter(etens),
	[&ctx](teq::TensptrT t)
	{
		return eteq::ETensor(t, ctx);
	});
	return etens;
}

distr::ox::TopographyT save_model (
	onnx::ModelProto& pb_model,
	const global::CfgMapptrT& ctx)
{
	teq::TensptrSetT uniques;

	eteq::ETensorsT etens;
	onnx::TensptrIdT identified;

	auto& graphinfo = eteq::get_graphinfo(ctx);
#ifdef ORDERED_SAVE
	types::StringsT ids;
	graphinfo.foreach(
	[&](teq::TensptrT tens, const std::string& id)
	{
		identified.insert({tens, id});
		ids.push_back(id);
	});
	std::sort(ids.begin(), ids.end());
	etens.reserve(ids.size());
	std::transform(ids.begin(), ids.end(),
		std::back_inserter(etens),
		[&graphinfo, &ctx](std::string id)
		{ return eteq::ETensor(graphinfo.get(id), ctx); });
#else
	graphinfo.foreach(
	[&](teq::TensptrT tens, const std::string& id)
	{
		identified.insert({tens, id});
		etens.push_back(eteq::ETensor(tens, ctx));
	});
#endif
	return save_model(pb_model, etens, identified);
}

eteq::ETensorsT load_model (
	global::CfgMapptrT& ctx,
	const onnx::ModelProto& pb_model,
	const distr::ox::TopographyT& topography)
{
	auto& graphinfo = eteq::get_graphinfo(ctx);

	onnx::TensptrIdT ids;
	teq::TensptrsT roots;
	if (auto mgr = get_distrmgr(ctx))
	{
		roots = distr::get_oxsvc(*mgr).load_graph(
			ids, pb_model.graph(), topography);
	}
	else
	{
		roots = serial::load_graph(ids, pb_model.graph());
	}

	// replace tensors mapped by id
	for (auto& idpair : ids)
	{
		if (teq::TensptrT src = graphinfo.get(idpair.right))
		{
			graphinfo.replace(src, idpair.left);
		}
	}

	eteq::ETensorsT out;
	out.reserve(roots.size());
	std::transform(roots.begin(), roots.end(),
		std::back_inserter(out),
		[&](teq::TensptrT tens)
		{
			return eteq::ETensor(tens, ctx);
		});
	return out;
}

}

#endif
