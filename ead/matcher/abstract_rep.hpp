#include "ead/generated/codes.hpp"

#include "ead/ead.hpp"
#include "ead/coord.hpp"
#include "ead/constant.hpp"
#include "ead/variable.hpp"

#ifndef EAD_ABSTRACT_REP_HPP
#define EAD_ABSTRACT_REP_HPP

namespace ead
{

enum OP_CLASS
{
	BAD_CLASS = 0,
	LEAF,
	UNARY, // unary op
	BINARY, // binary and not commutative
	NNARY, // nnary and commutative
	TRANSFORM,
	REDUCTION,
};

static std::unordered_map<
	age::_GENERATED_OPCODE,OP_CLASS> op_classes =
{
	{age::BAD_OP, BAD_CLASS},
	{age::ABS, UNARY},
	{age::NEG, UNARY},
	{age::SIN, UNARY},
	{age::COS, UNARY},
	{age::TAN, UNARY},
	{age::EXP, UNARY},
	{age::LOG, UNARY},
	{age::ROUND, UNARY},
	{age::SQRT, UNARY},
	{age::SQUARE, UNARY},
	{age::CUBE, UNARY},
	{age::SIGMOID, UNARY},
	{age::SIGMOID_GRAD, UNARY},
	{age::TANH, UNARY},
	{age::ADD, NNARY},
	{age::MUL, NNARY},
	{age::MAX, NNARY},
	{age::MIN, NNARY},
	{age::REDUCE_MAX, REDUCTION},
	{age::REDUCE_MIN, REDUCTION},
	{age::REDUCE_PROD, REDUCTION},
	{age::REDUCE_SUM, REDUCTION},
	{age::EXTEND, TRANSFORM},
	{age::PERMUTE, TRANSFORM},
	{age::POW, BINARY},
	{age::SUB, BINARY},
	{age::DIV, BINARY},
	{age::EQ, BINARY},
	{age::NEQ, BINARY},
	{age::GT, BINARY},
	{age::LT, BINARY},
	{age::RAND_UNIF, BINARY},
	{age::MATMUL, BINARY},
	{age::CONV, BINARY},
};

struct TransformLink
{
	age::_GENERATED_OPCODE transform_code_;

	ade::CoordptrT shaper_;

	CoordptrT coorder_;
};

struct AbstractRep;

using AbstractptrT = std::shared_ptr<AbstractRep>;

using TransformRepsT = std::list<TransformLink>;

struct ArgRep
{
	AbstractptrT arg_;

	ade::CoordptrT shaper_;

	CoordptrT coorder_;

	TransformRepsT transforms_;
};

using ArgRepsT = std::vector<ArgRep>;

using SmartLeafMapT = std::unordered_map<ade::iLeaf*,ade::TensptrT>;

struct AbstractRep
{
	virtual ~AbstractRep (void) = default;

	virtual ade::iLeaf* get_leaf (void) const = 0;

	virtual ArgRepsT get_args (void) const = 0;

	virtual age::_GENERATED_OPCODE get_opcode (void) const = 0;

	virtual OP_CLASS get_opclass (void) const = 0;
};

struct LeafRep final : public AbstractRep
{
	LeafRep (ade::iLeaf* leaf) : leaf_(leaf) {}

	ade::iLeaf* get_leaf (void) const override
	{
		return leaf_;
	}

	ArgRepsT get_args (void) const override
	{
		return {};
	}

	age::_GENERATED_OPCODE get_opcode (void) const override
	{
		return age::BAD_OP;
	}

	OP_CLASS get_opclass (void) const override
	{
		return LEAF;
	}

	ade::iLeaf* leaf_;
};

struct FuncRep final : public AbstractRep
{
	FuncRep (age::_GENERATED_OPCODE opcode, ArgRepsT args) :
		opcode_(opcode), args_(args) {}

	ade::iLeaf* get_leaf (void) const override
	{
		return nullptr;
	}

	ArgRepsT get_args (void) const override
	{
		return args_;
	}

	age::_GENERATED_OPCODE get_opcode (void) const override
	{
		return opcode_;
	}

	OP_CLASS get_opclass (void) const override
	{
		return op_classes[opcode_];
	}

	age::_GENERATED_OPCODE opcode_;

	ArgRepsT args_;
};

struct Abstracizer final : public ade::iTraveler
{
	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (abstracts_.end() == abstracts_.find(leaf))
		{
			abstracts_.emplace(leaf, ArgRep{
				std::make_shared<LeafRep>(leaf),
				ade::CoordptrT(),
				CoordptrT(),
				TransformRepsT()
			});
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (abstracts_.end() != abstracts_.find(func))
		{
			return;
		}
		age::_GENERATED_OPCODE opcode = (age::_GENERATED_OPCODE) func->get_opcode().code_;
		OP_CLASS oklass = op_classes[opcode];
		auto children = func->get_children();
		if (TRANSFORM == oklass)
		{
			auto tens = children[0].get_tensor();
			auto shaper = children[0].get_shaper();
			auto coorder = children[0].get_coorder();
			assert(nullptr != coorder);
			tens->accept(*this);

			auto arg = get_arg(tens);
			arg.transforms_.push_front(
				TransformLink{
					opcode,
					shaper,
					std::static_pointer_cast<CoordMap>(coorder)
				});
			abstracts_.emplace(func, arg);
			return;
		}
		std::list<ArgRep> args;
		for (auto& child : children)
		{
			auto tens = child.get_tensor();
			tens->accept(*this);
			auto arg = get_arg(tens);
			arg.shaper_ = child.get_shaper();
			arg.coorder_ = std::static_pointer_cast<CoordMap>(child.get_coorder());
			args.push_back(arg);
		}
		if (NNARY == oklass) // todo: move to optimizer
		{
			// merge with nnary arguments of the same op
			bool merged = false;
			ArgRepsT extras;
			for (auto it = args.begin(), et = args.end(); it != et;
				merged ? (it = args.erase(it)) : ++it)
			{
				auto& abs_node = it->arg_;
				merged = (opcode == abs_node->get_opcode());
				if (merged)
				{
					// merge with sub argument
					auto& transforms = it->transforms_;
					auto sub_args = abs_node->get_args();
					for (auto& sub_arg : sub_args)
					{
						auto fresh_transform = transforms;
						fresh_transform.insert(fresh_transform.end(),
							sub_arg.transforms_.begin(), sub_arg.transforms_.end());
						sub_arg.transforms_ = fresh_transform;
						extras.push_back(sub_arg);
					}
				}
			}
			args.insert(args.end(), extras.begin(), extras.end());
		}
		abstracts_.emplace(func, ArgRep{
			std::make_shared<FuncRep>(opcode, ArgRepsT(args.begin(), args.end())),
			ade::CoordptrT(),
			CoordptrT(),
			TransformRepsT()
		});
	}

	ArgRep& get_arg (ade::TensptrT& tens)
	{
		auto& arg = abstracts_[tens.get()];
		if (LEAF == arg.arg_->get_opclass() &&
			leaf_map_.end() == leaf_map_.find(arg.arg_->get_leaf()))
		{
			leaf_map_.emplace(arg.arg_->get_leaf(), tens);
		}
		return arg;
	}

	std::unordered_map<ade::iTensor*,ArgRep> abstracts_;

	SmartLeafMapT leaf_map_;
};

struct GraphRep
{
	ArgRepsT roots_;

	SmartLeafMapT leaves_;
};

GraphRep represent (ade::TensT tensors);

void optimize (GraphRep& graph);

template <typename T>
NodesT<T> actualize (const GraphRep& rep)
{
	const SmartLeafMapT& leaves = rep.leaves_;
	const ArgRepsT& roots = rep.roots_;
	NodesT<T> outs;
	outs.reserve(roots.size());
	for (const ArgRep& root : roots)
	{
		NodeptrT<T> out;
		if (LEAF == root.arg_->get_opclass())
		{
			out = to_node<T>(leaves.at(root.arg_->get_leaf()));
		}
		else
		{
			age::_GENERATED_OPCODE op = root.arg_->get_opcode();
			ArgRepsT args = root.arg_->get_args();
			GraphRep temp{
				args,
				leaves,
			};
			NodesT<T> children = actualize<T>(temp);
			ArgsT<T> ead_children;
			size_t nchildren = children.size();
			ead_children.reserve(nchildren);
			for (size_t i = 0; i < nchildren; ++i)
			{
				ead_children.push_back(FuncArg<T>(children[i],
					args[i].shaper_, args[i].coorder_));
			}

			if (NNARY == op_classes[op])
			{
				assert(nchildren > 1);
				out = ead::make_functor<T>(ade::Opcode{age::name_op(op), op}, {
					ead_children[0], ead_children[1]
				});
				for (size_t i = 2; i < nchildren; ++i)
				{
					out = ead::make_functor<T>(ade::Opcode{age::name_op(op), op}, {
						identity_map(out), ead_children[i]
					});
				}
			}
			else
			{
				out = ead::make_functor<T>(ade::Opcode{age::name_op(op), op}, ead_children);
			}
		}
		auto& transforms = root.transforms_;
		for (auto it = transforms.rbegin(), et = transforms.rend(); it != et; ++it)
		{
			out = ead::make_functor<T>(ade::Opcode{
				age::name_op(it->transform_code_), it->transform_code_
			}, {FuncArg<T>(out, it->shaper_, it->coorder_)});
		}
		outs.push_back(out);
	}
	return outs;
}

}

#endif // EAD_ABSTRACT_REP_HPP
