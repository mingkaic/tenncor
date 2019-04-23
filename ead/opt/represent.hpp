#include "ade/ifunctor.hpp"

#ifndef EAD_REPRESENT_HPP
#define EAD_REPRESENT_HPP

namespace ead
{

namespace opt
{

struct RuleContext;

struct iReprNode;

using RepptrT = std::shared_ptr<iReprNode>;

struct iRuleNode;

using RuleptrT = std::shared_ptr<iRuleNode>;

struct LeafRep;

struct FuncRep;

struct iReprNode
{
	virtual ~iReprNode (void) = default;

	virtual bool rulify (RuleContext& ctx, const RuleptrT& rule) = 0;
};

struct ReprArg final
{
	ReprArg (RepptrT arg,
		ade::CoordptrT shaper, ade::CoordptrT coorder) :
		arg_(arg), shaper_(shaper), coorder_(coorder)
	{
		if (nullptr == arg)
		{
			logs::fatal("created a representation argument with null argument");
		}
	}

	RepptrT arg_;

	ade::CoordptrT shaper_;

	ade::CoordptrT coorder_;
};

using RepArgsT = std::vector<ReprArg>;

struct iRuleNode
{
	virtual ~iRuleNode (void) = default;

	virtual bool process (RuleContext& ctx, LeafRep* leaf) const = 0;

	virtual bool process (RuleContext& ctx, FuncRep* func) const = 0;
};

struct RuleArg final
{
	RuleArg (RuleptrT arg,
		ade::CoordptrT shaper, ade::CoordptrT coorder) :
		arg_(arg), shaper_(shaper), coorder_(coorder)
	{
		if (nullptr == arg)
		{
			logs::fatal("created a rule argument with null argument");
		}
	}

	RuleptrT arg_;

	ade::CoordptrT shaper_;

	ade::CoordptrT coorder_;
};

using RuleArgsT = std::vector<RuleArg>;

struct LeafRep final : public iReprNode
{
	LeafRep (ade::iLeaf* leaf) : leaf_(leaf) {}

	bool rulify (RuleContext& ctx, const RuleptrT& rule) override
	{
		return rule.process(ctx, this);
	}

	ade::iLeaf* leaf_;
};

struct FuncRep final : public Representation
{
	FuncRep (ade::Opcode op, RepArgsT args) :
		op_(op), args_(args) {}

	bool rulify (RuleContext& ctx, const RuleptrT& rule) override
	{
		return rule.process(ctx, this);
	}

	ade::Opcode op_;

	RepArgsT args_;
};

struct Representer final : public ade::iTraveler
{
	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (reps_.end() == reps_.find(leaf))
		{
			reps_.emplace(leaf, std::make_shared<LeafRep>(leaf));
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (reps_.end() == reps_.find(func))
		{
			auto& children = func->get_children();
			std::vector<FuncRep::FuncArg> args;
			args.reserve(children.size());
			for (auto& child : children)
			{
				auto tens = child.get_tensor();
				tens->accept(*this);
				args.push_back(FuncRep::FuncArg{
					reps_[tens.get()],
					child.get_shaper(),
					child.get_coorder(),
				})
			}
			reps_.emplace(func,
				std::make_shared<FuncRep>(func->get_opcode(), args));
		}
	}

	std::unordered_map<ade::iTensor*,RepptrT> reps_;
};

template <typename T>
struct GraphAbstracter final
{
	void abstracize (ead::NodeptrT<T>& node)
	{
		auto tens = node->get_tensor();
		tens->accept(representer_);
		tens->accept(finder_);
		tens->accept(ider_);
	}

	// merge duplicate nodes

	// precalculate constants

private:
	Representer representer_;

	DistanceFinder finder_;

	Identifier<T> ider_;
};

}

}

#endif // EAD_REPRESENT_HPP
