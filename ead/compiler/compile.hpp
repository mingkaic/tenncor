#include <fstream>

#include <boost/dll/import.hpp>

#include "ade/functor.hpp"

#include "ead/generated/codes.hpp"

#include "ead/tensor.hpp"

#include "ead/compiler/plugins/plugin.hpp"

namespace ead
{

namespace compiler
{

const std::string impl_fmt =
"#include <boost/config.hpp> // for BOOST_SYMBOL_EXPORT\n"
"\n"
"#include \"ead/compiler/plugins/plugin.hpp\"\n"
"#include \"ead/random.hpp\"\n"
"\n"
"#ifndef PLUGIN_IMPL_HPP\n"
"#define PLUGIN_IMPL_HPP\n"
"\n"
"namespace compiler\n"
"{\n"
"\n"
"using PluginTypeT = %s;"
"\n"
"struct CompiledPlugin final : public iCompiledPlugin<PluginTypeT>\n"
"{\n"
"	ead::EigenptrT<PluginTypeT> calculate (size_t graph_id,\n"
"	   std::vector<ead::TensMapT<PluginTypeT>*> refs) override\n"
"	{\n"
"		switch (graph_id)\n"
"		{\n%s}\n"
"		return nullptr;\n"
"	}\n"
"};\n"
"\n"
"extern \"C\" BOOST_SYMBOL_EXPORT CompiledPlugin plugin;\n"
"CompiledPlugin plugin;\n"
"\n"
"}\n"
"\n"
"#endif // PLUGIN_IMPL_HPP\n";

const std::string compile_cmd =
"g++ --std=c++14 -Iexternal/com_github_eigenteam_eigen "
"-Iexternal/boost -I. %s -o %s -shared -fPIC";

const std::unordered_map<size_t,std::string> op_fmt =
{
	{age::ABS, "%s.abs()"},
	{age::NEG, "-%s"},
	{age::SIN, "%s.unaryExpr("
		"[](const PluginTypeT& a) -> PluginTypeT {"
		"	return std::sin(a);"
		"});"},
	{age::COS, "%s.unaryExpr("
		"[](const PluginTypeT& a) -> PluginTypeT {"
		"	return std::cos(a);"
		"});"},
	{age::TAN, "%s.unaryExpr("
		"[](const PluginTypeT& a) -> PluginTypeT {"
		"	return std::tan(a);"
		"});"},
	{age::EXP, "%s.exp()"},
	{age::LOG, "%s.log()"},
	{age::SQRT, "%s.sqrt()"},
	{age::SQUARE, "%s.square()"},
	{age::CUBE, "%s.cube()"},
	{age::ROUND, "%s.round()"},
	{age::SIGMOID, "%s.sigmoid()"},
	{age::TANH, "%s.tanh()"},
	{age::REDUCE_SUM, "%s.sum(%s).reshape(ead::DimensionsT{%s})"},
	{age::REDUCE_PROD, "%s.prod(%s).reshape(ead::DimensionsT{%s})"},
	{age::REDUCE_MIN, "%s.minimum(%s).reshape(ead::DimensionsT{%s})"},
	{age::REDUCE_MAX, "%s.maximum(%s).reshape(ead::DimensionsT{%s})"},
	{age::EXTEND, "%s.broadcast(%s).reshape(ead::DimensionsT{%s})"},
	{age::PERMUTE, "%s.shuffle(%s)"},
	{age::POW, "%s.binaryExpr(%s,\n"
		"[](const PluginTypeT& a, const PluginTypeT& b) -> PluginTypeT {\n"
		"	return std::pow(a, b);\n"
		"})"},
	{age::ADD, "%s+%s"},
	{age::SUB, "%s-%s"},
	{age::MUL, "%s*%s"},
	{age::DIV, "%s/%s"},
	{age::MIN, "%s.cwiseMin(%s)"},
	{age::MAX, "%s.cwiseMax(%s)"},
	{age::EQ, "%s.binaryExpr(%s,\n"
		"[](const PluginTypeT& a, const PluginTypeT& b) -> PluginTypeT {\n"
		"	return a == b;\n"
		"})"},
	{age::NEQ, "%s.binaryExpr(%s,\n"
		"[](const PluginTypeT& a, const PluginTypeT& b) -> PluginTypeT {\n"
		"	return a != b;\n"
		"})"},
	{age::LT, "%s.binaryExpr(%s,\n"
		"[](const PluginTypeT& a, const PluginTypeT& b) -> PluginTypeT {\n"
		"	return a < b;\n"
		"})"},
	{age::GT, "%s.binaryExpr(%s,\n"
		"[](const PluginTypeT& a, const PluginTypeT& b) -> PluginTypeT {\n"
		"	return a > b;\n"
		"})"},
	{age::RAND_UNIF, "%s.binaryExpr(%s, unif<PluginTypeT>)"},
};

inline std::string str_raw_coord (const ade::FuncArg& arg)
{
	ade::CoordT coord;
	arg.get_coorder()->forward(coord.begin(), coord.begin());
	return fmts::sprintf("std::array<uint8_t,%d>{", ade::rank_cap) +
		fmts::join(",", coord.begin(), coord.end()) + "}";
}

inline std::string str_red_coord (const ade::FuncArg& arg)
{
	ade::CoordT coord;
	arg.get_coorder()->forward(coord.begin(), coord.begin());
	std::vector<ade::DimT> vdims;
	std::copy_if(coord.begin(), coord.end(), std::back_inserter(vdims),
		[](ade::DimT d) { return d < ade::rank_cap; });
	return fmts::sprintf("std::array<uint8_t,%d>{", vdims.size()) +
		fmts::join(",", vdims.begin(), vdims.end()) + "}";
}

#define _INTERNAL_NAME_TYPE(REALTYPE)return #REALTYPE;

template <typename T>
std::string type_str (void)
{
	TYPE_LOOKUP(_INTERNAL_NAME_TYPE, age::get_type<T>())
}

template <typename T>
using PluginptrT = boost::shared_ptr<::compiler::iCompiledPlugin<T>>;

template <typename T>
using EigenMapT = std::unordered_map<ade::iTensor*,EigenptrT<T>>;

template <typename T>
struct CompiledOut final
{
	PluginptrT<T> _;
	EigenMapT<T> bridges_;
};

struct GraphStat final : public ade::iTraveler
{
	void visit (ade::iLeaf* leaf) override
	{
		graphsize_.emplace(leaf, 0);
	}

	void visit (ade::iFunctor* func) override
	{
		if (treat_as_leaf_.end() != treat_as_leaf_.find(func))
		{
			graphsize_.emplace(func, 0);
		}
		else if (graphsize_.end() == graphsize_.find(func))
		{
			ade::ArgsT children = func->get_children();
			size_t ngraph = 0;
			for (auto& child : children)
			{
				ade::iTensor* tens = child.get_tensor().get();
				tens->accept(*this);
				auto childinfo = graphsize_.find(tens);
				if (graphsize_.end() != childinfo &&
					childinfo->second > ngraph)
				{
					ngraph = childinfo->second;
				}
			}
			graphsize_[func] = ngraph + 1;
		}
	}

	std::unordered_set<ade::iFunctor*> treat_as_leaf_;

	std::unordered_map<ade::iTensor*,size_t> graphsize_;
};

template <typename T>
std::vector<ade::iTensor*> defn_case (std::ostream& stmts,
	std::unordered_set<ade::iFunctor*>& treat_as_leaf, ade::iFunctor* bridge)
{
	GraphStat stat;
	stat.treat_as_leaf_ = treat_as_leaf;
	bridge->accept(stat);
	treat_as_leaf.emplace(bridge);

	// store functors and leaves
	std::vector<ade::iTensor*> ordered(stat.graphsize_.size());
	std::transform(stat.graphsize_.begin(), stat.graphsize_.end(),
		ordered.begin(),
		[](std::pair<ade::iTensor*,size_t> graphinfo)
		{
			return graphinfo.first;
		});
	std::sort(ordered.begin(), ordered.end(),
		[&](ade::iTensor* a, ade::iTensor* b)
		{
			return stat.graphsize_[a] < stat.graphsize_[b];
		});

	std::vector<ade::iTensor*> leaves;
	std::vector<ade::iFunctor*> funcs;
	{
		// gather the leaves
		auto it = ordered.begin(), et = ordered.end();
		for (; it != et && stat.graphsize_[*it] == 0; ++it)
		{
			leaves.push_back(*it);
		}
		// gather the functors
		for (; it != et; ++it)
		{
			funcs.push_back(
				static_cast<ade::iFunctor*>(*it));
		}
	}

	// create statements
	// maps nodes to its variable name
	std::unordered_map<ade::iTensor*,std::string> varnames;
	for (size_t i = 0, n = leaves.size(); i < n; ++i)
	{
		std::string varname = fmts::sprintf("leaf_%d", i);
		varnames.emplace(leaves[i], varname);
		stmts << "auto& " << varname << " = *refs[" << i << "];\n";
	}

	// maps nodes being bridged to their respective occurrence
	std::unordered_map<ade::iTensor*,size_t> bridges;
	size_t nfuncs = funcs.size();
	for (size_t i = 0; i < nfuncs; ++i)
	{
		ade::iFunctor* func = funcs[i];
		std::string varname = fmts::sprintf("func_%d", i);
		varnames.emplace(func, varname);
		const auto& children = func->get_children();
		size_t nchildren = children.size();
		std::vector<ade::iTensor*> variables(nchildren);
		std::vector<std::string> params(nchildren);
		for (size_t j = 0; j < nchildren; ++j)
		{
			ade::iTensor* tens = children[j].get_tensor().get();
			variables[j] = tens;
			params[j] = varnames[tens];
		}

		std::string decl;
		stmts << "auto " << varname << " = ";
		size_t opcode = func->get_opcode().code_;
		switch (opcode)
		{
			case age::ABS:
			case age::NEG:
			case age::SIN:
			case age::COS:
			case age::TAN:
			case age::EXP:
			case age::LOG:
			case age::SQRT:
			case age::SQUARE:
			case age::CUBE:
			case age::ROUND:
			case age::SIGMOID:
			case age::TANH:
				stmts << fmts::sprintf(op_fmt.at(opcode), params[0].c_str()) << ";\n";
				break;
			case age::REDUCE_SUM:
			case age::REDUCE_PROD:
			case age::REDUCE_MIN:
			case age::REDUCE_MAX:
			{
				ade::Shape shape = func->shape();
				stmts << fmts::sprintf(op_fmt.at(opcode), params[0].c_str(),
					str_red_coord(func->get_children()[0]).c_str(),
					fmts::join(",", shape.begin(), shape.end()).c_str()) << ";\n";
			}
				break;
			case age::EXTEND:
			{
				ade::Shape shape = func->shape();
				stmts << fmts::sprintf(op_fmt.at(opcode), params[0].c_str(),
					str_raw_coord(func->get_children()[0]).c_str(),
					fmts::join(",", shape.begin(), shape.end()).c_str()) << ";\n";
			}
				break;
			case age::PERMUTE:
				stmts << fmts::sprintf(op_fmt.at(opcode), params[0].c_str(),
					str_raw_coord(func->get_children()[0]).c_str()) << ";\n";
				break;
			case age::POW:
			case age::ADD:
			case age::SUB:
			case age::MUL:
			case age::DIV:
			case age::MIN:
			case age::MAX:
			case age::EQ:
			case age::NEQ:
			case age::LT:
			case age::GT:
			case age::RAND_UNIF:
				stmts << fmts::sprintf(op_fmt.at(opcode),
					params[0].c_str(), params[1].c_str()) << ";\n";
				break;
			default:
				logs::fatalf("unknown operation %s",
					func->get_opcode().name_.c_str());
		}
	}
	ade::Shape shape = bridge->shape();
	stmts << "return ead::make_eigentensor<PluginTypeT>({"
		<< fmts::join(",", shape.begin(), shape.end())
		<< "}, " << varnames[bridge] << ");\n";

	return leaves;
}

template <typename T>
CompiledOut<T> compile_roots (NodesT<T> roots)
{
	std::vector<ade::iFunctor*> bridges;
	{
		ade::GraphStat stat;
		for (NodeptrT<T>& root : roots)
		{
			auto root_tens = root->get_tensor().get();
			root_tens->accept(stat);
			if (stat.graphsize_[root_tens] > 0)
			{
				bridges.push_back(
					static_cast<ade::iFunctor*>(root_tens));
			}
		}

		for (auto& graphinfo : stat.graphsize_)
		{
			if (graphinfo.second > 0)
			{
				auto f = static_cast<ade::iFunctor*>(graphinfo.first);
				// bridges surround matrix operations (only MATMUL so far)
				if (f->get_opcode().code_ == age::MATMUL)
				{
					bridges.push_back(f);
				}
			}
		}
		std::sort(bridges.begin(), bridges.end(),
			[&](ade::iTensor* a, ade::iTensor* b)
			{
				return stat.graphsize_[a] < stat.graphsize_[b];
			});
		bridges.erase(std::unique(bridges.begin(), bridges.end()), bridges.end());
	}

	EigenMapT<T> bridge_assoc; // map real itensor node to generic bridge

	std::vector<ade::iTensor*> build_order;
	std::unordered_map<ade::iTensor*,
		std::function<EigenptrT<T>(PluginptrT<T>&)>> build_graph_ref;

	size_t graph_id = 0;
	std::stringstream stmts;
	std::unordered_set<ade::iFunctor*> treat_as_leaf;
	for (size_t i = 0, n = bridges.size(); i < n; ++i)
	{
		ade::iFunctor* bridge = bridges[i];
		// if bridge is a matrix operation (MATMUL so far)
		// define a case for all of bridge's arguments
		if (bridge->get_opcode().code_ == age::MATMUL)
		{
			const auto& children = bridge->get_children();
			for (const auto& child : children)
			{
				ade::iTensor* tens = child.get_tensor().get();
				auto f = dynamic_cast<ade::iFunctor*>(tens);
				if (nullptr != f &&
					build_graph_ref.end() == build_graph_ref.find(tens))
				{
					stmts << "case " << graph_id << ": {\n";
					auto leaves = defn_case<T>(stmts, treat_as_leaf, f);
					stmts << "}\n";
					build_order.push_back(tens);
					build_graph_ref.emplace(tens,
						[&bridge_assoc, graph_id, leaves](PluginptrT<T>& plugin)
						{
							std::vector<ead::TensMapT<T>*> leaf_refs(
								leaves.size());
							std::transform(leaves.begin(),leaves.end(),
								leaf_refs.begin(),
								[&](ade::iTensor* tens)
								{
									auto it = bridge_assoc.find(tens);
									if (bridge_assoc.end() == it)
									{
										return static_cast<ead::iLeaf<T>*>(
											tens)->get_tensmap();
									}
									return &it->second->get_tensmap();
								});
							return plugin->calculate(graph_id, leaf_refs);
						});
					++graph_id;
				}
			}
			build_order.push_back(bridge);
			build_graph_ref.emplace(bridge,
				[&bridge_assoc, bridge](PluginptrT<T>& plugin)
				{
					ade::Shape shape = bridge->shape();
					const auto& children = bridge->get_children();
					std::vector<ead::TensMapT<T>*> leaf_refs(
						children.size());
					std::transform(children.begin(),children.end(),
						leaf_refs.begin(),
						[&](const ade::FuncArg& arg)
						{
							ade::iTensor* tens = arg.get_tensor().get();
							auto it = bridge_assoc.find(tens);
							if (bridge_assoc.end() == it)
							{
								return static_cast<ead::iLeaf<T>*>(
									tens)->get_tensmap();
							}
							return &it->second->get_tensmap();
						});
					return ead::make_eigenmatrix<T>(shape_convert(shape),
						ead::tensmap_to_matmap(*leaf_refs[0]) *
						ead::tensmap_to_matmap(*leaf_refs[1]));
				});
			treat_as_leaf.emplace(bridge);
		}
		else // bridge is a node
		{
			stmts << "case " << graph_id << ": {\n";
			auto leaves = defn_case<T>(stmts, treat_as_leaf, bridge);
			stmts << "}\n";
			build_order.push_back(bridge);
			build_graph_ref.emplace(bridge,
				[&bridge_assoc, graph_id, leaves](PluginptrT<T>& plugin)
				{
					std::vector<ead::TensMapT<T>*> leaf_refs(
						leaves.size());
					std::transform(leaves.begin(),leaves.end(),
						leaf_refs.begin(),
						[&](ade::iTensor* tens)
						{
							auto it = bridge_assoc.find(tens);
							if (bridge_assoc.end() == it)
							{
								return static_cast<ead::iLeaf<T>*>(
									tens)->get_tensmap();
							}
							return &it->second->get_tensmap();
						});
					return plugin->calculate(graph_id, leaf_refs);
				});
			++graph_id;
		}
	}

	std::string impl_filepath = "/tmp/compiled_plugin.cc";
	std::string impl_libpath = "/tmp/compiled_plugin";

	std::ofstream fout(impl_filepath);
	fout << fmts::sprintf(impl_fmt, type_str<T>().c_str(),
		stmts.str().c_str()) << std::endl;
	fout.close();

	std::system(fmts::sprintf(compile_cmd,
		impl_filepath.c_str(), impl_libpath.c_str()).c_str());

	PluginptrT<T> plugin = boost::dll::import<::compiler::iCompiledPlugin<T>>(
		impl_libpath, "plugin", boost::dll::load_mode::append_decorations);

	for (ade::iTensor* bridge : build_order)
	{
		auto eigen_ref = build_graph_ref[bridge](plugin);
		bridge_assoc.emplace(bridge, eigen_ref);
	}
	return CompiledOut<T>{plugin, bridge_assoc};
}

}

template <typename T>
using UpdatersT = std::list<EigenptrT<T>>;

template <typename T>
UpdatersT<T> order_updates (compiler::CompiledOut<T>& compiled_out)
{
	auto& bridges = compiled_out.bridges_;
	std::list<ade::iTensor*> order;
	ade::GraphStat stat;
	for (auto& graph : bridges)
	{
		graph.first->accept(stat);
		order.push_back(graph.first);
	}
	order.sort([&](ade::iTensor* a, ade::iTensor* b)
		{
			return stat.graphsize_[a] < stat.graphsize_[b];
		});
	UpdatersT<T> out(order.size());
	std::transform(order.begin(), order.end(), out.begin(),
		[&](ade::iTensor* tens)
		{
			return bridges[tens];
		});
	return out;
}

}
