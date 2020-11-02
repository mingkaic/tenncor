#include "tenncor/serial.hpp"

#ifdef TENNCOR_SERIAL_HPP

namespace tcr
{

const std::string app_name = "tenncor";
const std::string app_version = "1.0.0";
const std::string tenncor_dom = "com.mingkaic.tenncor";

distr::ox::TopographyT save_model (
	onnx::ModelProto& pb_model,
	const eteq::ETensorsT& roots,
	const onnx::TensptrIdT& identified)
{
	pb_model.set_ir_version(onnx::IR_VERSION);
	pb_model.set_producer_name(app_name);
	pb_model.set_producer_version(app_version);
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

}

#endif
