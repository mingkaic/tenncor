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

void save_graph (cortenn::Graph& out, teq::TensptrsT roots,
	tag::TagRegistry& registry = tag::get_reg());

void load_graph (teq::TensptrSetT& roots,
	const cortenn::Graph& pb_graph,
	tag::TagRegistry& registry = tag::get_reg());

static std::vector<double> convert_attrs (const marsh::Maps& attrs, std::string key)
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
		teq::ShaperT shaper = nullptr;
		eigen::CoordptrT coorder = nullptr;

		auto shape_vals = convert_attrs(edge.second, eigen::shaper_key);
		auto coord_vals = convert_attrs(edge.second, eigen::coorder_key);

		if (shape_vals.size() > 0)
		{
			shaper = std::make_shared<teq::ShapeMap>(
				[&](teq::MatrixT& arg)
				{
					for (size_t i = 0, n = shape_vals.size(); i < n; ++i)
					{
						arg[i / teq::mat_dim][i % teq::mat_dim] = shape_vals[i];
					}
				});
		}

		if (coord_vals.size() > 0)
		{
			coorder = std::make_shared<eigen::CoordMap>(
				[&](teq::MatrixT& arg)
				{
					for (size_t i = 0, n = coord_vals.size(); i < n; ++i)
					{
						arg[i / teq::mat_dim][i % teq::mat_dim] = coord_vals[i];
					}
				});
		}

		tmp_edges.push_back(
			FuncArg<T>(to_node<T>(edge.first), shaper, coorder));
	}
	return teq::TensptrT(Functor<T>::get(
		teq::Opcode{opname, egen::get_op(opname)},tmp_edges));
}

}

#endif // ETEQ_SERIALIZE_HPP
