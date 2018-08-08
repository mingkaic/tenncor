#include <cstring>
#include <algorithm>

#include "sand/operator.hpp"

#include "soil/data.hpp"
#include "soil/functor.hpp"
#include "soil/grader.hpp"
#include "soil/constant.hpp"

#include "util/error.hpp"

#ifdef SOIL_FUNCTOR_HPP

static inline Meta metamorph (
	std::vector<Nodeptr>& args, iPreOperator& preop)
{
	std::vector<Meta> infos(args.size());
	std::transform(args.begin(), args.end(), infos.begin(),
	[](Nodeptr& arg)
	{
		return Meta{arg->shape(), arg->type()};
	});
	return preop(infos);
}

Nodeptr Functor::get (std::vector<Nodeptr> args,
	iPreOperator& preop, OPCODE opcode)
{
	return Nodeptr(new Functor(args, preop, opcode));
}

std::shared_ptr<char> Functor::calculate (Pool& pool)
{
	if (has_op(opcode_, info_.type_))
	{
		std::vector<std::shared_ptr<char> > temp; // todo: remove with pool implementation
		std::vector<NodeInfo> args;
		std::transform(args_.begin(), args_.end(), std::back_inserter(args),
		[&pool, &temp](Nodeptr& arg)
		{
			temp.push_back(arg->calculate(pool));
			return NodeInfo{
				temp.back().get(),
				arg->shape()
			};
		});
		std::shared_ptr<char> out = make_data(info_.nbytes());
		NodeInfo dest{
			out.get(),
			info_.shape_
		};
		get_op(opcode_, info_.type_)(dest, args, encoder_.data_);
		return out;
	}
	else if (args_.size() != 1)
	{
		handle_error("cannot resolve multiple args without aggregating op");
	}
	return args_[0]->calculate(pool);
}

Nodeptr Functor::gradient (Nodeptr& leaf) const
{
	if (leaf.get() == this)
	{
		return get_one(info_);
	}
	return get_grader(opcode_)(args_, leaf);
}

std::vector<iNode*> Functor::get_refs (void) const
{
	std::vector<iNode*> out(args_.size());
	std::transform(args_.begin(), args_.end(), out.begin(),
	[](const Nodeptr& arg)
	{
		return arg.get();
	});
	return out;
}

Functor::Functor (std::vector<Nodeptr> args,
	iPreOperator& preop, OPCODE opcode) :
	Node(metamorph(args, preop)),
	args_(args), encoder_(preop.encode()), opcode_(opcode) {}

#endif
