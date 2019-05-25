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

/// Tree node for labeling Tensptrs
struct PathedTens final
{
	~PathedTens (void)
	{
		for (auto& child : children_)
		{
			delete child.second;
		}
	}

	/// Grab all leaf and branch nodes from subtree root other
	/// Accounting for duplicate labels but not Tensptrs
	void join (PathedTens* other)
	{
		{
			std::vector<std::string> labels;
			for (auto opair : other->tens_)
			{
				if (tens_.end() != tens_.find(opair.first))
				{
					labels.push_back(opair.first);
				}
			}
			if (tens_.size() > 0)
			{
				logs::warnf("duplicate base labels %s",
					fmts::to_string(labels.begin(), labels.end()).c_str());
			}
		}

		for (auto cpair : other->children_)
		{
			std::string label = cpair.first;
			auto it = children_.find(label);
			if (children_.end() == it)
			{
				children_.emplace(label, cpair.second);
			}
			else
			{
				it->second->join(cpair.second);
			}
		}
	}

	/// Return tensor associated with input path if found otherwise nullptr
	ade::TensptrT get_labelled (StringsT path) const
	{
		return get_labelled(path.begin(), path.end());
	}

	/// Set input path to reference tensor
	void set_labelled (StringsT path, ade::TensptrT tens)
	{
		set_labelled(path.begin(), path.end(), tens);
	}

	/// Return tensor associated with path between iterators begin and end
	/// if found otherwise nullptr
	ade::TensptrT get_labelled (
		StringsT::iterator path_begin,
		StringsT::iterator path_end) const
	{
		if (path_begin == path_end)
		{
			return nullptr;
		}
		auto path_it = path_begin++;
		if (path_begin == path_end)
		{
			auto it = tens_.find(*path_it);
			if (tens_.end() != it)
			{
				return it->second;
			}
		}
		else
		{
			auto it = children_.find(*path_it);
			if (nullptr != it->second)
			{
				return it->second->get_labelled(path_begin, path_end);
			}
		}
		return nullptr;
	}

	/// Set path between iterators begin and end to reference tensor
	void set_labelled (StringsT::iterator path_begin,
		StringsT::iterator path_end, ade::TensptrT tens)
	{
		if (path_begin == path_end)
		{
			return;
		}
		std::string label = *(path_begin++);
		if (path_begin == path_end)
		{
			tens_.emplace(label, tens);
			return;
		}
		PathedTens* child;
		auto it = children_.find(label);
		if (it == children_.end())
		{
			child = new PathedTens();
			children_.emplace(label, child);
		}
		else
		{
			assert(nullptr != it->second);
			child = it->second;
		}
		child->set_labelled(path_begin, path_end, tens);
	}

	/// Map of labels to branching nodes
	std::unordered_map<std::string,PathedTens*> children_;

	/// Map of labels to tensor leaves
	std::unordered_map<std::string,ade::TensptrT> tens_;
};

/// Contains all information necessary to recreate labelled ADE graph
struct GraphInfo final
{
	/// Set of all roots (Tensptrs without any parent)
	std::unordered_set<ade::TensptrT> roots_;

	/// Labelled tensors
	PathedTens tens_;
};

/// Return graph info through out available from in graph
template <typename LOAD, typename std::enable_if<std::is_base_of<iLoader,LOAD>::value>::type* = nullptr>
void load_graph (GraphInfo& out, const cortenn::Graph& in)
{
	LOAD loader;
	auto nodes = in.nodes();
	TensT invec;
	for (const cortenn::Node& node : nodes)
	{
		auto pb_labels = node.labels();
		if (node.has_source())
		{
			std::string src_label;
			if (pb_labels.size() > 0)
			{
				src_label = *(pb_labels.rbegin());
			}
			const cortenn::Source& source = node.source();
			std::string sstr = source.shape();
			ade::Shape shape(std::vector<ade::DimT>(sstr.begin(), sstr.end()));
			std::string data = source.data();
			ade::TensptrT leaf = loader.generate_leaf(data.c_str(),
				shape, source.typelabel(), src_label, source.is_const());
			invec.push_back(leaf);
			if (false == pb_labels.empty())
			{
				StringsT labels(pb_labels.begin(), pb_labels.end());
				out.tens_.set_labelled(labels.begin(), labels.end(), leaf);
			}
			out.roots_.emplace(leaf);
		}
		else
		{
			cortenn::Functor func = node.functor();
			auto nodeargs = func.args();
			std::string opname = func.opname();
			ade::Opcode opcode{opname, age::get_op(opname)};
			ade::ArgsT args;
			for (auto nodearg : nodeargs)
			{
				ade::TensptrT arg = invec[nodearg.idx()];
				auto shaper_pb = nodearg.shaper();
				auto coorder_pb = nodearg.coord();
				std::vector<double> shaper_vec(shaper_pb.begin(), shaper_pb.end());
				std::vector<double> coord_vec(coorder_pb.begin(), coorder_pb.end());
				ade::CoordptrT shaper = loader.generate_shaper(shaper_vec);
				ade::CoordptrT coord = loader.generate_coorder(opcode, coord_vec);
				args.push_back(
					ade::FuncArg(arg, shaper, nodearg.fwd(), coord));
				out.roots_.erase(invec[nodearg.idx()]);
			}
			ade::TensptrT f = loader.generate_func(opcode, args);
			invec.push_back(f);
			if (false == pb_labels.empty())
			{
				StringsT labels(pb_labels.begin(), pb_labels.end());
				out.tens_.set_labelled(labels.begin(), labels.end(), f);
			}
			out.roots_.emplace(f);
		}
	}
}

}

#endif // PBM_GRAPH_HPP
