///
/// serialize.hpp
/// eteq
///
/// Purpose:
/// Define functions for marshal and unmarshal data sources
///

#include "eigen/generated/opcode.hpp"
#include "eigen/generated/dtype.hpp"

#include "eteq/constant.hpp"
#include "eteq/variable.hpp"
#include "eteq/functor.hpp"

#include "pbm/save.hpp"
#include "pbm/load.hpp"

#ifndef ETEQ_SERIALIZE_HPP
#define ETEQ_SERIALIZE_HPP

namespace eteq
{

pbm::TensMapIndicesT save_graph (
	tenncor::Graph& out, teq::TensptrsT roots,
	tag::TagRegistry& registry = tag::get_reg());

void load_graph (teq::TensptrSetT& roots,
	const tenncor::Graph& pb_graph,
	tag::TagRegistry& registry = tag::get_reg());

static inline std::vector<double> convert_attrs (
	const marsh::Maps& attrs, std::string key)
{
	std::vector<double> out;
	if (estd::has(attrs.contents_, key))
	{
		const auto& objs = attrs.contents_.at(key);
		if (typeid(marsh::NumArray<double>).hash_code() == objs->class_code())
		{
			out = static_cast<const marsh::NumArray<double>*>(
				objs.get())->contents_;
		}
	}
	return out;
}

template <typename T>
static teq::TensptrT convert_func (
	std::string opname, const pbm::EdgesT& edges)
{
	ArgsT<T> tmp_edges;
	tmp_edges.reserve(edges.size());
	for (auto& edge : edges)
	{
		teq::Shape shape = edge.first->shape();
		auto shape_vals = convert_attrs(edge.second, eigen::shaper_key);
		if (shape_vals.size() > 0)
		{
			shape = teq::Shape(std::vector<teq::DimT>(
				shape_vals.begin(), shape_vals.end()));
		}
		std::vector<double> coords = convert_attrs(edge.second,
			eigen::coorder_key);

		tmp_edges.push_back(
			Edge<T>(to_node<T>(edge.first), shape, coords));
	}
	return make_functor<T>(egen::get_op(opname),tmp_edges)->get_tensor();
}

}

#endif // ETEQ_SERIALIZE_HPP
