#include "ead/generated/codes.hpp"

#include "ead/ead.hpp"
#include "ead/coord.hpp"
#include "ead/constant.hpp"
#include "ead/variable.hpp"

#ifndef EAD_REPRESENT_HPP
#define EAD_REPRESENT_HPP

namespace ead
{

namespace opt
{

struct Representation;

using RepptrT = std::shared_ptr<Representation>;

struct Representation
{
	virtual ~Representation (void) = default;

	virtual RepptrT rulify (size_t depth) const = 0;
};

struct RuleLeafRep final : public Representation
{
	RuleLeafRep (Representation* rep) : rep_(rep) {}

	RepptrT rulify (size_t depth) const override
	{
		return rep_->rulify(depth);
	}

	Representation* rep_;
};

struct LeafRep final : public Representation
{
	LeafRep (ade::iLeaf* leaf) : leaf_(leaf) {}

	RepptrT rulify (size_t depth) const override
	{
		if (depth < 1)
		{
			return std::make_shared<RuleLeafRep>(this);
		}
		return std::make_shared<LeafRep>(this->leaf_);
	}

	ade::iLeaf leaf_;
};

struct FuncRep final : public Representation
{
	struct FuncArg
	{
		RepptrT arg_;

		ade::CoordptrT shaper_;

		ade::CoordptrT coorder_;
	};

	FuncRep (ade::Opcode op, std::vector<FuncArg> args) :
		op_(op), args_(args) {}

	RepptrT rulify (size_t depth) const override
	{
		if (depth < 1)
		{
			return std::make_shared<RuleLeafRep>(this);
		}
		std::vector<FuncArg> fresh_args;
		fresh_args.reserve(args_.size());
		for (FuncArg arg : args_)
		{
			fresh_args.push_back(FuncArg{
				arg.arg_->rulify(depth - 1),
				arg.shaper_,
				arg.coorder_,
			});
		}
		return std::make_shared<FuncRep>(op_, fresh_args);
	}

	ade::Opcode op_;

	std::vector<FuncArg> args_;
};

struct Representer : public ade::iTraveler
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


const std::string args_begin = "(";
const std::string args_end = ")";
const std::string args_delim = ",";
const std::string shape_delim = "->";
const std::string coord_delim = "_";

std::string pid (const void* addr)
{
	std::stringstream ss;
	ss << std::hex << (size_t) addr;
	return ss.str();
}

template <typename GENERATOR>
std::string uuid (GENERATOR& rgen, const void* addr)
{
	static std::uniform_int_distribution<size_t> tok_dist(0, 15);
	auto now = std::chrono::system_clock::now();
	std::time_t now_c = std::chrono::system_clock::to_time_t(now);

	std::stringstream ss;
	ss << std::hex << now_c << (size_t) addr;

	for (size_t i = 0; i < 16; i++)
	{
		size_t token = tok_dist(rgen);
		ss << std::hex << token;
	}
	return ss.str();
}

template <typename T>
struct Stringifier final : public ade::iTraveler
{
	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (reps_.end() == reps_.find(leaf))
		{
			if (static_cast<iLeaf<T>*>(leaf)->is_const())
			{
				T* data = (T*) leaf->data();
				size_t n = leaf->shape().n_elems();
				if (std::all_of(data + 1, data + n,
					[&data](T val) { return val == *data; }))
				{
					reps_.emplace(leaf, fmts::to_string(*data));
					return;
				}
			}
			reps_.emplace(leaf, pid(leaf));
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (reps_.end() == reps_.find(func))
		{
			auto children = func->get_children();
			std::vector<std::string> args;
			args.reserve(children.size());
			for (auto& child : children)
			{
				auto tens = child.get_tensor();
				tens->accept(*this);
				std::string arg_rep = reps_[tens.get()];

				auto shaper = child.get_shaper();
				auto coorder = child.get_coorder();
				if (false == ade::is_identity(shaper.get()) ||
					false == ade::is_identity(coorder.get()))
				{
					arg_rep = trans_encode(
						arg_rep, shaper, coorder);
				}
				args.push_back(arg_rep);
			}
			std::string opname = func->get_opcode().name_;
			reps_.emplace(func, opname + args_begin +
				fmts::join(args_delim, args.begin(), args.end()) + args_end);
		}
	}

	std::string trans_encode (std::string arg_rep,
		const ade::CoordptrT& shaper, const ade::CoordptrT& coorder)
	{
		std::string shaper_id = pid(shaper.get());
		shapers_.emplace(shaper_id, shaper);

		std::stringstream ss;
		ss << arg_rep << shape_delim << shaper_id << coord_delim;
		if (coorder)
		{
			ss << coorder->to_string();
		}
		else
		{
			ss << fmts::arr_begin + fmts::arr_end;
		}
		return ss.str();
	}

	std::unordered_map<ade::iTensor*,std::string> reps_;

	std::unordered_map<std::string,ade::CoordptrT> shapers_;
};

}

}

#endif // EAD_REPRESENT_HPP
