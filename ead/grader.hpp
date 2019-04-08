///
/// grader.hpp
/// bwd
///
/// Purpose:
/// Define grader traveler to build partial derivative equations
///

#include <list>

#include "ead/generated/api.hpp"
#include "ead/generated/grader.hpp"

#include "ead/constant.hpp"
#include "ead/edge.hpp"

#ifndef EAD_GRADER_HPP
#define EAD_GRADER_HPP

namespace ead
{

struct CoordInfo
{
	CoordptrT fwd_;

	CoordptrT rev_;
};

struct Transform
{
	template <typename T>
	NodeptrT<T> apply (NodeptrT<T> fwd) const
	{
		return make_functor<T>(op_, {
			FuncArg<T>(fwd, shaper_, coorder_.fwd_)
		});
	}

	template <typename T>
	NodeptrT<T> rev_apply (const NodeptrT<T>& fwd) const
	{
		ade::CoordptrT shaper(shaper_->reverse());
		const CoordptrT& coorder = coorder_.rev_;
		ade::Opcode opcode;
		switch (op_.code_)
		{
			case age::REDUCE_SUM:
				opcode = ade::Opcode{
					"EXTEND",
					age::EXTEND,
				};
				break;
			case age::EXTEND:
				opcode = ade::Opcode{
					"REDUCE_SUM",
					age::REDUCE_SUM,
				};
				break;
			case age::PERMUTE:
				opcode = ade::Opcode{
					"PERMUTE",
					age::PERMUTE,
				};
				break;
			default:
				logs::fatalf("unknown transform: %s... ignoring it",
					op_.name_.c_str());
		}
		return make_functor<T>(opcode, {FuncArg<T>(fwd, shaper, coorder)});
	}

	ade::Opcode op_ = {"", age::BAD_OP};

	ade::CoordptrT shaper_;

	CoordInfo coorder_;
};

// ordered according to transformation-order
using TransformsT = std::list<Transform>;

template <typename T>
NodeptrT<T> apply (const TransformsT& transforms, NodeptrT<T> fwd)
{
	NodeptrT<T> out = fwd;
	for (const Transform& transform : transforms)
	{
		out = transform.apply(out);
	}
	return out;
}

// apply reversal of these transformations
template <typename T>
NodeptrT<T> rev_apply (const TransformsT& transforms, NodeptrT<T> fwd)
{
	NodeptrT<T> out = fwd;
	for (auto it = transforms.rbegin(), et = transforms.rend(); it != et; ++it)
	{
		out = it->rev_apply(out);
	}
	return out;
}

CoordptrT reverse_reduce (ade::Shape shape,
	const ade::CoordptrT& coorder)
{
	ade::CoordT dims;
	coorder->forward(dims.begin(), dims.begin());

	ade::CoordT bcast;
	std::fill(bcast.begin(), bcast.end(), 1);
	for (uint8_t d : dims)
	{
		if (d < ade::rank_cap)
		{
			bcast[d] = shape.at(d);
		}
	}
	return std::make_shared<CoordMap>(EXTEND, bcast, false);
}

CoordptrT reverse_extend (const ade::CoordptrT& coorder)
{
	ade::CoordT dims;
	coorder->forward(dims.begin(), dims.begin());

	std::vector<uint8_t> red_dims;
	for (uint8_t i = 0; i < ade::rank_cap; ++i)
	{
		if (dims[i] > 1)
		{
			red_dims.push_back(i);
		}
	}
	return reduce(red_dims);
}

CoordptrT reverse_permute (const ade::CoordptrT& coorder)
{
	ade::CoordT dims;
	coorder->forward(dims.begin(), dims.begin());

	ade::CoordT order;
	for (uint8_t i = 0; i < ade::rank_cap; ++i)
	{
		order[dims[i]] = i;
	}
	return std::make_shared<CoordMap>(PERMUTE, order, true);
}

template <typename T>
struct Jacobian
{
	Jacobian (NodeptrT<T> jac) : jac_(jac) {}

	Jacobian (NodeptrT<T> jac,
		TransformsT& reds, TransformsT& adjs) :
		jac_(jac), reds_(reds), adjs_(adjs) {}

	// node holding the most shape-information before reducing to wrt-shape
	NodeptrT<T> jac_;

	// list of Transforms to reduce jac_ to wrt-shape
	TransformsT reds_; // aka red-shaper

	// list of Transforms to adjacent operations to fit jac_
	// example:
	//	A:shape<a, b> - reduce -> B:shape<a>
	//	B:shape<a> * C:shape<a> -> D:shape<a>
	//	dA/dA has jac_ shape <a, b>
	//	dB/dA has jac_ shape <a, b> since <a, b> has more info than <a>
	//	dD/dA = dB/dA * C, since C has shape <a>,
	//	C needs to fit to <a, b> using dB/dA's adjs_
	TransformsT adjs_; // aka adj-shaper
};

// derive bottom-up (the way automatic-diff should be)
template <typename T>
NodeptrT<T> derive_bu (EdgesT& edges,
	ade::TensptrT root, ade::TensptrT wrt) // todo: replace derive
{
	ade::PathFinder finder(wrt.get());
	root->accept(finder);

	auto& pathmap = finder.parents_;
	// no path to wrt
	if (pathmap.empty())
	{
		return make_constant_scalar((T) 0, wrt->shape());
	}
	// else there exists a path to wrt
	// using pathfinder, breadth first traverse from this to wrt
	std::string target_str;
	if (auto target = dynamic_cast<Variable<T>*>(wrt.get()))
	{
		target_str = target->label_;
	}
	else
	{
		target_str = wrt->to_string();
	}
	ade::GraphStat stat;
	ade::OwnerTracker tracker;
	root->accept(stat);
	root->accept(tracker);
	tracker.owners_.emplace(root.get(), root);

	std::list<ade::iFunctor*> parents;
	std::transform(pathmap.begin(), pathmap.end(),
		std::back_inserter(parents),
		[](std::pair<ade::iTensor*,std::unordered_set<size_t>> parent)
		{
			return static_cast<ade::iFunctor*>(parent.first);
		});
	// parents go from wrt to root
	parents.sort(
		[&](ade::iFunctor* a, ade::iFunctor* b)
		{
			return stat.graphsize_[a] < stat.graphsize_[b];
		});

	// todo: reuse this
	auto wrt_grad = make_constant_scalar((T) 1, wrt->shape());
	std::unordered_map<const ade::iTensor*,Jacobian<T>> jacobians =
		{{wrt.get(), Jacobian<T>(wrt_grad)}};
	edges.push_back(Edge{wrt, wrt_grad->get_tensor(), ade::Opcode{
		fmts::sprintf("GRADIENT_%s", target_str.c_str()), GRADIENT}});

	for (ade::iFunctor* parent : parents)
	{
		auto children = parent->get_children();
		auto& child_indices = pathmap[parent];
		size_t nchild_indices = child_indices.size();
		assert(nchild_indices > 0);
		std::vector<Jacobian<T>> jac_args;
		jac_args.reserve(nchild_indices);
		for (size_t child_idx : child_indices)
		{
			Jacobian<T> jacobian = jacobians.at(children[child_idx].get_tensor().get());
			NodeptrT<T> grad = nullptr;
			switch (parent->get_opcode().code_)
			{
				// REGULAR OP -> boring cases...
				case age::ABS:
				{
					auto a = to_node<T>(children[child_idx].get_tensor());
					grad = age::div(a, age::abs(a));
				}
					break;
				case age::NEG:
					grad = make_constant_scalar<T>(-1, children[child_idx].shape());
					break;
				case age::SIN:
				{
					auto a = to_node<T>(children[child_idx].get_tensor());
					grad = age::cos(a);
				}
					break;
				case age::COS:
				{
					auto a = to_node<T>(children[child_idx].get_tensor());
					grad = age::neg(age::sin(a));
				}
					break;
				case age::TAN:
				{
					auto a = to_node<T>(children[child_idx].get_tensor());
					grad = age::pow(age::cos(a), make_constant_scalar<T>(-2, a->shape()));
				}
					break;
				case age::EXP:
					grad = to_node<T>(tracker.owners_[parent].lock());
					break;
				case age::LOG:
				{
					auto a = to_node<T>(children[child_idx].get_tensor());
					grad = age::div(make_constant_scalar<T>(1, a->shape()), a);
				}
					break;
				case age::SQRT:
				{
					auto a = to_node<T>(children[child_idx].get_tensor());
					grad = age::div(make_constant_scalar<T>(1, a->shape()),
						age::mul(make_constant_scalar<T>(2, a->shape()), age::sqrt(a)));
				}
					break;
				case age::SQUARE:
				{
					auto a = to_node<T>(children[child_idx].get_tensor());
					grad = age::mul(make_constant_scalar<T>(2, a->shape()), a);
				}
					break;
				case age::CUBE:
				{
					auto a = to_node<T>(children[child_idx].get_tensor());
					grad = age::mul(make_constant_scalar<T>(3, a->shape()), age::square(a));
				}
					break;
				case age::SIGMOID:
				{
					auto a = to_node<T>(children[child_idx].get_tensor());
					grad = age::sigmoid_grad(a);
				}
					break;
				case age::SIGMOID_GRAD:
				{
					auto a = to_node<T>(children[child_idx].get_tensor());
					grad = age::mul(to_node<T>(tracker.owners_[parent].lock()),
						age::sub(make_constant_scalar<T>(1, a->shape()),
							age::mul(make_constant_scalar<T>(2, a->shape()), age::sigmoid(a))
						));
				}
					break;
				case age::TANH:
				{
					auto a = to_node<T>(children[child_idx].get_tensor());
					grad = age::sub(make_constant_scalar<T>(1, a->shape()),
						age::square(to_node<T>(tracker.owners_[parent].lock())));
				}
					break;
				case age::ROUND:
				case age::ADD:
					grad = make_constant_scalar<T>(1, children[child_idx].shape());
					break;
				case age::MUL:
					grad = to_node<T>(children[(size_t)(child_idx == 0)].get_tensor());
					break;
				case age::MAX:
				case age::MIN:
				{
					auto a = to_node<T>(children[child_idx].get_tensor());
					grad = age::eq(to_node<T>(tracker.owners_[parent].lock()), a);
				}
					break;
				case age::POW:
				{
					auto a = to_node<T>(children[0].get_tensor());
					auto b = to_node<T>(children[1].get_tensor());
					grad = child_idx == 0 ?
						age::mul(b, age::pow(a, age::sub(b, make_constant_scalar<T>(1, b->shape())))) :
						age::mul(age::log(a), to_node<T>(tracker.owners_[parent].lock()));
				}
					break;
				case age::SUB:
					grad = make_constant_scalar<T>(
						child_idx == 0 ? 1 : -1, children[child_idx].shape());
					break;
				case age::DIV:
				{
					auto a = to_node<T>(children[0].get_tensor());
					auto b = to_node<T>(children[1].get_tensor());
					grad = child_idx == 0 ?
						age::div(make_constant_scalar<T>(1, b->shape()), b) :
						age::div(age::div(age::neg(a), b), b);
				}
					break;
				case age::EQ:
				case age::NEQ:
				case age::GT:
				case age::LT:
				case age::RAND_UNIF:
					grad = make_constant_scalar<T>(0, children[child_idx].shape());
					break;
				// REDUCE_* -> JAC: *, adj: EXTEND, red: _
				case age::REDUCE_SUM:
					// JAC = JAC
					jacobian.adjs_.push_front(Transform{
						ade::Opcode{"EXTEND", age::EXTEND},
						ade::CoordptrT(children[child_idx].get_shaper()->reverse()),
						CoordInfo{
							reverse_reduce(children[child_idx].get_tensor()->shape(),
								children[child_idx].get_coorder()),
							std::static_pointer_cast<CoordMap>(children[child_idx].get_coorder()),
						}
					});
					break;
				case age::REDUCE_PROD:
				{
					// JAC = JAC * EXTEND(REDUCE_PROD(X)) / X
					Transform adj{
						ade::Opcode{"EXTEND", age::EXTEND},
						ade::CoordptrT(children[child_idx].get_shaper()->reverse()),
						CoordInfo{
							reverse_reduce(children[child_idx].get_tensor()->shape(),
								children[child_idx].get_coorder()),
							std::static_pointer_cast<CoordMap>(children[child_idx].get_coorder()),
						}
					};
					jacobian.jac_ = age::mul(apply<T>(jacobian.adjs_,
						age::div(adj.apply(to_node<T>(tracker.owners_[parent].lock())),
							to_node<T>(children[child_idx].get_tensor()))), jacobian.jac_);
					jacobian.adjs_.push_front(adj);
				}
					break;
				case age::REDUCE_MAX:
				case age::REDUCE_MIN:
				{
					Transform adj{
						ade::Opcode{"EXTEND", age::EXTEND},
						ade::CoordptrT(children[child_idx].get_shaper()->reverse()),
						CoordInfo{
							reverse_reduce(children[child_idx].get_tensor()->shape(),
								children[child_idx].get_coorder()),
							std::static_pointer_cast<CoordMap>(children[child_idx].get_coorder()),
						}
					};
					// for * IN [MAX, MIN]
					// REDUCE_*_GRAD(X) = dX.JAC * EXTEND(REDUCE_*(X)) == X
					jacobian.jac_ = age::mul(apply<T>(jacobian.adjs_,
						age::eq(adj.apply(to_node<T>(tracker.owners_[parent].lock())),
							to_node<T>(children[child_idx].get_tensor()))), jacobian.jac_);
					jacobian.adjs_.push_front(adj);
				}
					break;
				// EXTEND(arg) -> JAC: EXTEND(JAC), adj: _, red: REDUCE_SUM
				case age::EXTEND:
					jacobian.jac_ = make_functor<T>(ade::Opcode{"EXTEND", age::EXTEND}, {FuncArg<T>(
						jacobian.jac_, children[child_idx].get_shaper(),
						std::static_pointer_cast<CoordMap>(children[child_idx].get_coorder())
					)});
					jacobian.reds_.push_front(Transform{
						ade::Opcode{"REDUCE_SUM", age::REDUCE_SUM},
						ade::CoordptrT(children[child_idx].get_shaper()->reverse()),
						CoordInfo{
							reverse_extend(children[child_idx].get_coorder()),
							std::static_pointer_cast<CoordMap>(children[child_idx].get_coorder()),
						}
					});
					break;
				// PERMUTE(arg) -> JAC: JAC, adj: reverse(PERMUTE), red: _
				case age::PERMUTE:
					jacobian.adjs_.push_front(Transform{
						ade::Opcode{"PERMUTE", age::PERMUTE},
						children[child_idx].get_shaper(),
						CoordInfo{
							reverse_permute(children[child_idx].get_coorder()),
							std::static_pointer_cast<CoordMap>(children[child_idx].get_coorder()),
						}
					});
					break;
				// PARTIAL_MATMUL(I) = I == 0 ? [EXTEND_B0, PERMUTE_210] : [EXTEND_A1, PERMUTE_021]
				// MATMUL -> JAC: PARTIAL_MATMUL(child_idx)(reverse(adjs)(JAC)) *
				//		PARTIAL_MATMUL(not child_idx)(child_idx ? a : b)),
				//	adj: EXTEND, red: reverse(PARTIAL_MATMUL(child_idx))
				case age::MATMUL:
				{
					TransformsT reds;
					NodeptrT<T> partial_opposition;
					ade::DimT common_dim = children[0].get_tensor()->shape().at(0);
					ade::DimT left_dim = children[1].get_tensor()->shape().at(0);
					ade::DimT right_dim = children[0].get_tensor()->shape().at(1);
					if (0 == child_idx)
					{
						reds = {
							Transform{
								ade::Opcode{"PERMUTE", age::PERMUTE},
								ade::permute({2,1,0}),
								CoordInfo{
									permute({2,1,0}),
									permute({2,1,0}),
								}
							},
							Transform{
								ade::Opcode{"REDUCE_SUM", age::REDUCE_SUM},
								ade::reduce(2, {left_dim}),
								CoordInfo{
									reduce({2}),
									extend(2, {left_dim}),
								}
							}
						};
						partial_opposition = age::permute(
							age::extend(to_node<T>(children[1].get_tensor()),
							2, {right_dim}), {0,2,1});
					}
					else
					{
						reds = {
							Transform{
								ade::Opcode{"PERMUTE", age::PERMUTE},
								ade::permute({0,2,1}),
								CoordInfo{
									permute({0,2,1}),
									permute({0,2,1}),
								}
							},
							Transform{
								ade::Opcode{"REDUCE_SUM", age::REDUCE_SUM},
								ade::reduce(2, {right_dim}),
								CoordInfo{
									reduce({2}),
									extend(2, {right_dim}),
								}
							}
						};
						partial_opposition = age::permute(
							age::extend(to_node<T>(children[0].get_tensor()),
							2, {left_dim}), {2,1,0});
					}

					// prev_adjs: <previous_jac_shape> -> <parent_shape>
					auto real_grad = rev_apply(jacobian.adjs_, jacobian.jac_);
					// reds: <parent_shape> -> <new_jac_shape>
					jacobian.jac_ = age::mul(rev_apply(reds, real_grad), partial_opposition);
					reds.insert(reds.end(), jacobian.reds_.begin(), jacobian.reds_.end());
					jacobian.reds_ = reds;
					jacobian.adjs_ = {Transform{
						ade::Opcode{"EXTEND", age::EXTEND},
						ade::extend(2, {common_dim}),
						CoordInfo{
							extend(2, {common_dim}),
							reduce({2}),
						}
					}};
				}
					break;
				case age::CONV:
				default:
					throw std::bad_function_call();
			}
			if (nullptr != grad)
			{
				jacobian.jac_ = age::mul(apply<T>(jacobian.adjs_, grad), jacobian.jac_);
			}
			jac_args.push_back(jacobian);
		}
		// assert for all jacobian in jac_args, jacobian.jac_ has same shapes
		Jacobian<T>& out = jac_args[0];
		for (size_t i = 1; i < nchild_indices; ++i)
		{
			Jacobian<T>& jacobian = jac_args[i];
			out.jac_ = age::add(out.jac_, jacobian.jac_);
			// minimize # of transformations
			if (out.reds_.size() > jacobian.reds_.size())
			{
				out.reds_ = jacobian.reds_;
			}
			if (out.adjs_.size() > jacobian.adjs_.size())
			{
				out.adjs_ = jacobian.adjs_;
			}
		}

		jacobians.emplace(parent, out);
		edges.push_back(Edge{
			tracker.owners_[parent],
			out.jac_->get_tensor(), ade::Opcode{
				fmts::sprintf("JACOBIAN_%s", target_str.c_str()), JACOBIAN}});
	}

	Jacobian<T>& root_jac = jacobians.at(root.get());
	auto root_grad = apply<T>(root_jac.reds_, root_jac.jac_);
	edges.push_back(Edge{root, root_grad->get_tensor(), ade::Opcode{
		fmts::sprintf("GRADIENT_%s", target_str.c_str()), GRADIENT}});
	return root_grad;
}

/// Traveler to obtain derivative of accepted node with respect to target
template <typename T>
struct Grader final : public ade::iTraveler
{
	Grader (const ade::iTensor* target) :
		target_(target)
	{
		if (target_ == nullptr)
		{
			logs::fatal("cannot derive with respect to null");
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		derivatives_.emplace(leaf,
			make_constant_scalar((T) (leaf == target_), target_->shape()));
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (func == target_)
		{
			derivatives_.emplace(func, make_constant_scalar((T) 1, target_->shape()));
			return;
		}

		ade::PathFinder finder(target_);
		func->accept(finder);

		auto& pathmap = finder.parents_;
		// no path to wrt
		if (pathmap.empty())
		{
			derivatives_.emplace(func, make_constant_scalar((T) 0, target_->shape()));
			return;
		}
		// else there exists a path to wrt
		// using pathfinder, breadth first traverse from this to wrt
		ade::GraphStat stat;
		func->accept(stat);

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

		std::unordered_map<const ade::iTensor*,NodesT<T>> grads = {
			{func, {make_constant_scalar((T) 1, func->shape())}},
		};
		for (ade::iFunctor* parent : parents)
		{
			NodesT<T>& gargs = grads[parent];
			NodeptrT<T> bwd = gargs[0];
			for (size_t i = 1, n = gargs.size(); i < n; ++i)
			{
				bwd = age::add(bwd, gargs[i]);
			}

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
				auto grad_step = age::chain_rule<T>(parent, bwd, args, i);
				grads[args[i].get()].push_back(grad_step);
				edges_.push_back(Edge{
					args[i],
					grad_step->get_tensor(),
					ade::Opcode{
						"GRADIENT",
						GRADIENT
					}
				});
			}
		}
		NodesT<T>& finalgargs = grads[target_];
		NodeptrT<T> finalgarg = finalgargs[0];
		for (size_t i = 1, n = finalgargs.size(); i < n; ++i)
		{
			finalgarg = age::add(finalgarg, finalgargs[i]);
		}
		derivatives_.emplace(func, finalgarg);
	}

	EdgesT edges_;

	/// Target of tensor all visited nodes are derived with respect to
	const ade::iTensor* target_;

	/// Map forward root node to derivative root
	std::unordered_map<const ade::iTensor*,NodeptrT<T>> derivatives_;
};

/// Derive root with respect to target and optimized
template <typename T>
NodeptrT<T> derive (NodeptrT<T> root, NodeptrT<T> target)
{
	auto rooten = root->get_tensor();
	Grader<T> grader(target->get_tensor().get());
	rooten->accept(grader);
	auto it = grader.derivatives_.find(rooten.get());
	assert(grader.derivatives_.end() != it);
	return it->second;
}

/// Derive root with respect to target and optimized
template <typename T>
NodeptrT<T> derive_with_edges (EdgesT& edges, NodeptrT<T> root, NodeptrT<T> target)
{
	auto rooten = root->get_tensor();
	Grader<T> grader(target->get_tensor().get());
	rooten->accept(grader);
	auto it = grader.derivatives_.find(rooten.get());
	assert(grader.derivatives_.end() != it);
	edges.insert(edges.end(),
		grader.edges_.begin(), grader.edges_.end());
	return it->second;
}

}

#endif // EAD_GRADER_HPP
