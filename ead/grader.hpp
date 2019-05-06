///
/// grader.hpp
/// bwd
///
/// Purpose:
/// Define grader traveler to build partial derivative equations
///

#include <list>

#include "ade/edge.hpp"

#include "ead/generated/api.hpp"
#include "ead/generated/grader.hpp"

#include "ead/constant.hpp"

#ifndef EAD_GRADER_HPP
#define EAD_GRADER_HPP

namespace ead
{

enum EDGE_CODE
{
	GRADIENT = 0,
};

/// Derive root with respect to target and optimized
template <typename T>
NodeptrT<T> derive (NodeptrT<T> root, NodeptrT<T> target)
{
	ade::EdgesT edges;
	return derive_with_edges(edges, root, target);
}

// ruler of chains
template <typename T>
struct ChainRuler
{
	// return orig derived with respect to args[idx]
	NodeptrT<T> dlocal (ead::NodeptrT<T> orig, size_t idx)
	{
		auto f = static_cast<ade::iFunctor*>(orig->get_tensor().get());
		const ade::ArgsT& args = f->get_children();
		NodeptrT<T> out = nullptr;
		switch ((age::_GENERATED_OPCODE) f->get_opcode().code_)
		{
			case age::ABS:
				out = age::div(ead::to_node<T>(args[0].get_tensor()), orig);
				break;
			case age::NEG:
				out = ead::make_constant_scalar<T>(-1, args[0].get_tensor()->shape());
				break;
			case age::SIN:
				out = age::cos(ead::to_node<T>(args[0].get_tensor()));
				break;
			case age::COS:
				out = age::neg(age::sin(ead::to_node<T>(args[0].get_tensor())));
				break;
			case age::TAN:
				out = age::div(
					ead::make_constant_scalar<T>(1,args[0].get_tensor()->shape()),
					age::pow(
						age::cos(ead::to_node<T>(args[0].get_tensor())),
						ead::make_constant_scalar<T>(2, args[0].get_tensor()->shape())
					)
				);
				break;
			case age::EXP:
				out = orig;
				break;
			case age::LOG:
				out = age::div(
					ead::make_constant_scalar<T>(1, args[0].get_tensor()->shape()),
					ead::to_node<T>(args[0].get_tensor())
				);
				break;
			case age::SQRT:
				out = age::div(
					ead::make_constant_scalar<T>(1, args[0].get_tensor()->shape()),
					age::mul(
						ead::make_constant_scalar<T>(2,args[0].get_tensor()->shape()),
						orig
					)
				);
				break;
			case age::SQUARE:
				out = age::mul(
					ead::make_constant_scalar<T>(2,args[0].get_tensor()->shape()),
					ead::to_node<T>(args[0].get_tensor())
				);
				break;
			case age::CUBE:
				out = age::mul(
					ead::make_constant_scalar<T>(3, args[0].get_tensor()->shape()),
					age::square(ead::to_node<T>(args[0].get_tensor()))
				);
				break;
			case age::ROUND:
				out = ead::make_constant_scalar<T>(1, args[0].get_tensor()->shape());
				break;
			case age::SIGMOID:
				out = age::sigmoid_grad(ead::to_node<T>(args[0].get_tensor()));
				break;
			case age::SIGMOID_GRAD:
				out = age::mul(
					orig,
					age::sub(
						ead::make_constant_scalar<T>(1,args[0].get_tensor()->shape()),
						age::mul(
							ead::make_constant_scalar<T>(2, args[0].get_tensor()->shape()),
							age::sigmoid(ead::to_node<T>(args[0].get_tensor()))
						)
					)
				);
				break;
			case age::TANH:
				out = age::sub(
					ead::make_constant_scalar<T>(1, args[0].get_tensor()->shape()),
					age::square(orig)
				);
				break;
			case age::ADD:
				out = ead::make_constant_scalar<T>(1, args[0].get_tensor()->shape());
				break;
			case age::MUL:
				out = ead::to_node<T>(args[(size_t)(idx==0)].get_tensor());
				break;
			case age::MAX:
			case age::MIN:
				out = age::eq(orig, ead::to_node<T>(args[idx].get_tensor()));
				break;
			case age::POW:
				out = idx==0 ?
					age::mul(
						ead::to_node<T>(args[1].get_tensor()),
						age::pow(
							ead::to_node<T>(args[0].get_tensor()),
							age::sub(
								ead::to_node<T>(args[1].get_tensor()),
								ead::make_constant_scalar<T>(1, args[0].get_tensor()->shape())
							)
						)
					) :
					age::mul(age::log(ead::to_node<T>(args[0].get_tensor())), orig);
				break;
			case age::SUB:
				out = ead::make_constant_scalar<T>(idx == 0 ? 1 : -1, args[0].get_tensor()->shape());
				break;
			case age::DIV:
			{
				auto denom = ead::to_node<T>(args[1].get_tensor());
				out = idx==0 ?
					age::div(
						ead::make_constant_scalar<T>(1, args[0].get_tensor()->shape()),
						denom
					) :
					age::div(
						age::div(age::neg(ead::to_node<T>(args[0].get_tensor())), denom),
						denom
					);
			}
				break;
			case age::EQ:
			case age::NEQ:
			case age::GT:
			case age::LT:
			case age::RAND_UNIF:
				out = ead::make_constant_scalar<T>(0, args[0].get_tensor()->shape());
				break;
			case age::REDUCE_SUM:
			case age::EXTEND:
			case age::PERMUTE:
				out = ead::make_constant_scalar<T>(1, args[0].get_tensor()->shape());
				break;
			case age::REDUCE_PROD:
				out = age::div(
					reduce_grad(args[0], orig, idx),
					to_node<T>(args[0].get_tensor())
				);
				break;
			case age::REDUCE_MAX:
			case age::REDUCE_MIN:
				out = age::eq(
					reduce_grad(args[0], orig, idx),
					to_node<T>(args[0].get_tensor())
				);
				break;
			case age::MATMUL:
			{
				NodeptrT<T> lhs = to_node<T>(args[0].get_tensor());
				NodeptrT<T> rhs = to_node<T>(args[1].get_tensor());
				out = 0 == idx ?
					// ext_rhs
					age::permute(age::extend(rhs, 2, {lhs->shape().at(1)}), {0,2,1}) :
					// ext_lhs
					age::permute(age::extend(lhs, 2, {rhs->shape().at(0)}), {2,1,0});
			}
				break;
			case age::CONV:
			{
				//
			}
				// break;
			default:
				logs::fatal("Unknown op");
		}
		return out;
	}

	NodeptrT<T> chain (NodeptrT<T> local, ade::iFunctor* fwd, NodeptrT<T> bwd, size_t idx)
	{
		NodeptrT<T> out;
		switch (fwd->get_opcode().code_)
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
			case age::SIGMOID_GRAD:
			case age::TANH:
			case age::ADD:
			case age::MUL:
			case age::MAX:
			case age::MIN:
			case age::POW:
			case age::SUB:
			case age::DIV:
			case age::EQ:
			case age::NEQ:
			case age::GT:
			case age::LT:
			case age::RAND_UNIF:
				out = age::mul(local, bwd);
				break;
			case age::REDUCE_MAX:
			case age::REDUCE_MIN:
			case age::REDUCE_PROD:
			case age::REDUCE_SUM:
				out = age::mul(local, reduce_grad(fwd->get_children()[0], bwd, idx));
				break;
			case age::EXTEND:
				out = age::mul(local, extend_grad(fwd, bwd, idx));
				break;
			case age::PERMUTE:
				out = age::mul(local, permute_grad(fwd, bwd, idx));
				break;
			case age::MATMUL:
				out = age::reduce_sum(
					age::permute(
						age::mul(local,
							age::extend(bwd, 2, {
								fwd->get_children()[0].get_tensor()->shape().at(0)
							})),
						0 == idx ?
							std::vector<uint8_t>{2, 1, 0} :
							std::vector<uint8_t>{0, 2, 1}), 2);
				break;
			case age::CONV:
			{
				//
			}
				// break;
			default:
				logs::fatal("Unknown op");
		}
		return out;
	}
};

/// Derive root with respect to target and optimized
template <typename T>
NodeptrT<T> derive_with_edges (ade::EdgesT& edges, NodeptrT<T> root, NodeptrT<T> target)
{
	if (root->get_tensor() == target->get_tensor())
	{
		return make_constant_scalar((T) 1, target->shape());
	}

	ade::iTensor* target_tens = target->get_tensor().get();
	ade::PathFinder finder(target_tens);
	root->get_tensor()->accept(finder);

	auto& pathmap = finder.parents_;
	// no path to wrt
	if (pathmap.empty())
	{
		return make_constant_scalar((T) 0, target->shape());
	}
	// else there exists a path to wrt
	// using pathfinder, breadth first traverse from this to wrt
	std::string source_str = root->get_tensor()->to_string();
	ade::GraphStat stat;
	root->get_tensor()->accept(stat);
	auto owners = ade::track_owners(root->get_tensor());

	std::list<ade::iFunctor*> parents;
	std::transform(pathmap.begin(), pathmap.end(),
		std::back_inserter(parents),
		[](std::pair<ade::iTensor*,std::unordered_set<size_t>> parent)
		{
			return static_cast<ade::iFunctor*>(parent.first);
		});
	parents.sort(
		[&](ade::iFunctor* a, ade::iFunctor* b)
		{
			return stat.graphsize_[a] > stat.graphsize_[b];
		});

	auto root_grad = make_constant_scalar((T) 1, root->shape());
	std::unordered_map<const ade::iTensor*,NodesT<T>> grads = {
		{root->get_tensor().get(), {root_grad}}
	};
	ChainRuler<T> ruler;
	for (ade::iFunctor* parent : parents)
	{
		NodesT<T>& gargs = grads[parent];
		NodeptrT<T> bwd = gargs[0];
		for (size_t i = 1, n = gargs.size(); i < n; ++i)
		{
			bwd = age::add(bwd, gargs[i]);
		}
		edges.push_back(ade::Edge{
			bwd->get_tensor(),
			owners[parent],
			ade::Opcode{
				fmts::sprintf("GRADIENT_%s_%s",
					source_str.c_str(), parent->to_string().c_str()),
				GRADIENT
			}
		});

		auto& grad_indices = pathmap[parent];
		ade::ArgsT children = parent->get_children();
		size_t nchildren = children.size();
		// assert: all nnary-children use identity mapping,
		// so no children-arg is direct mapping
		ade::TensT args(nchildren);
		std::transform(children.begin(), children.end(), args.begin(),
			[](ade::FuncArg& arg)
			{
				return arg.get_tensor();
			});
		// for each painted child, calculate dThis/dChild
		// go through grads in order
		std::list<size_t> ordered(grad_indices.begin(), grad_indices.end());
		ordered.sort();
		for (size_t i : ordered)
		{
			auto local = ruler.dlocal(ead::to_node<T>(owners[parent].lock()), i); // todo: store this for reuse
			edges.push_back(ade::Edge{
				local->get_tensor(),
				owners[parent],
				ade::Opcode{
					fmts::sprintf("GRADIENT_%s_%s",
						parent->to_string().c_str(), args[i]->to_string().c_str()),
					GRADIENT
				}
			});
			auto grad_step = ruler.chain(local, parent, bwd, i);
			// auto grad_step = age::chain_rule<T>(parent, bwd, args, i);
			grads[args[i].get()].push_back(grad_step);
		}
	}
	NodesT<T>& outargs = grads[target_tens];
	NodeptrT<T> out = outargs[0];
	for (size_t i = 1, n = outargs.size(); i < n; ++i)
	{
		out = age::add(out, outargs[i]);
	}
	std::string target_str;
	if (auto target_v = dynamic_cast<Variable<T>*>(target_tens))
	{
		target_str = target_v->label_;
	}
	else
	{
		target_str = target_tens->to_string();
	}
	edges.push_back(ade::Edge{
		out->get_tensor(),
		target->get_tensor(),
		ade::Opcode{
			fmts::sprintf("GRADIENT_%s_%s",
				source_str.c_str(), target_str.c_str()),
			GRADIENT
		}
	});
	return out;
}

}

#endif // EAD_GRADER_HPP
