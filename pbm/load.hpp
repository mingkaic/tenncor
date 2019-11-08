///
/// graph.hpp
/// pbm
///
/// Purpose:
/// Define functions for marshal and unmarshal equation graph
///

#include "pbm/data.hpp"

#ifndef PBM_LOAD_HPP
#define PBM_LOAD_HPP

namespace pbm
{

/// Return graph info through out available from in graph
template <typename LOAD, typename std::enable_if<
	std::is_base_of<iLoader,LOAD>::value>::type* = nullptr>
void load_graph (teq::TensptrSetT& out, const cortenn::Graph& in)
{
	LOAD loader;
	auto nodes = in.nodes();
	TensptrsT invec;
	for (const cortenn::Node& node : nodes)
	{
		teq::TensptrT tens;
		if (node.has_source())
		{
			const cortenn::Source& source = node.source();
			auto& slist = source.shape();
			teq::Shape shape(std::vector<teq::DimT>(slist.begin(), slist.end()));
			std::string data = source.data();
			teq::TensptrT leaf = loader.generate_leaf(data.c_str(),
				shape, source.typelabel(), node.label(), source.is_const());
			invec.push_back(leaf);
			tens = leaf;
		}
		else
		{
			cortenn::Functor func = node.functor();
			auto nodeargs = func.args();
			std::string opname = func.opname();
			teq::ArgsT args;
			for (auto nodearg : nodeargs)
			{
				teq::TensptrT arg = invec[nodearg.idx()];
				auto shaper_pb = nodearg.shaper();
				auto coorder_pb = nodearg.coord();
				std::vector<double> shaper_vec(shaper_pb.begin(), shaper_pb.end());
				std::vector<double> coord_vec(coorder_pb.begin(), coorder_pb.end());
				teq::ShaperT shaper = loader.generate_shaper(shaper_vec);
				teq::CvrtptrT coord = loader.generate_coorder(opname, coord_vec);
				args.push_back(
					teq::FuncArg(arg, shaper, nodearg.fwd(), coord));
				out.erase(invec[nodearg.idx()]);
			}
			teq::TensptrT f = loader.generate_func(opname, args);
			invec.push_back(f);
			tens = f;
		}
		auto& pb_tags = node.tags();
		for (auto& tagpair : pb_tags)
		{
			const std::string& tagkey = tagpair.first;
			auto& taglabels = tagpair.second.labels();
			auto tagr = tag::get_reg().tagr_by_key(tagkey);
			for (std::string taglabel : taglabels)
			{
				tagr(tens, taglabel);
			}
		}
		out.emplace(tens);
	}
}

}

#endif // PBM_GRAPH_HPP
