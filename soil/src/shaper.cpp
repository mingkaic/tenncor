#include <cassert>

#include "soil/shaper.hpp"
#include "soil/error.hpp"
#include "soil/mapper.hpp"

#ifdef SHAPER_HPP

using Shaper = std::function<Shape(std::vector<iNode*>)>;

Shape elem_shaper (std::vector<iNode*> args)
{
	if (args.size() == 0)
	{
		handle_error("creating elementary shape not from no shapes",
			ErrArg<size_t>{"num_args", args.size()});
	}

	Shape outshape = args.front()->shape();
	uint8_t outrank = outshape.n_rank();
	for (auto it = args.begin() + 1, et = args.end(); it != et; ++it)
	{
		Shape shape = (*it)->shape();
		uint8_t rank = shape.n_rank();
		if (false == outshape.compatible_before(shape, std::min(outrank, rank)))
		{
			handle_error("incompatible elem shapes",
				ErrArg<std::string>{"first_shape", outshape.to_string()},
				ErrArg<std::string>{"cur_shape", shape.to_string()});
		}
		if (rank > outrank)
		{
			outshape = shape;
			outrank = rank;
		}
	}
	return outshape;
}

Shape transpose_shaper (std::vector<iNode*> args)
{
	if (args.size() != 1)
	{
		handle_error("creating tranpose shape from multiple shapes",
			ErrArg<size_t>{"num_args", args.size()});
	}

	Shape shape = args.front()->shape();
	Shape shape0 = shape.group(0);
	Shape shape1 = shape.group(1);
	std::vector<Shape> groups = shape.as_groups();
	if (groups.empty())
	{
		groups = {shape1, shape0};
	}
	else if (groups.size() == 1)
	{
		groups[0] = shape1;
		groups.push_back(shape0);
	}
	else
	{
		groups[0] = shape1;
		groups[1] = shape0;
	}

	return Shape(groups);
}

Shape matmul_shaper (std::vector<iNode*> args)
{
	if (args.size() != 2)
	{
		handle_error("creating matmul shape not from 2 shapes",
			ErrArg<size_t>{"num_args", args.size()});
	}

	Shape ashape = args.front()->shape();
	Shape bshape = args.back()->shape();
	if (false == ashape.group(0).compatible_after(bshape.group(1), 0))
	{
		handle_error("incompatible common dimension in matmul",
			ErrArg<std::string>{"ashape", ashape.to_string()},
			ErrArg<std::string>{"bshape", bshape.to_string()});
	}
	Shape shape0 = bshape.group(0);
	Shape shape1 = ashape.group(1);
	uint8_t adim = shape1.n_rank();
	uint8_t bdim = shape0.n_rank();
	uint8_t cdim = ashape.group(0).n_rank();
	uint8_t arank = ashape.n_rank();
	uint8_t groupranka = cdim + adim;
	auto ait = ashape.begin();
	if (false == std::equal(ait + std::min(groupranka, arank),
		ait + arank, bshape.begin() + bdim + cdim))
	{
		handle_error("incompatible dimensions beyond first 2 groups in matmul",
			ErrArg<std::string>{"ashape", ashape.to_string()},
			ErrArg<std::string>{"bshape", bshape.to_string()});
	}

	std::vector<Shape> groups = ashape.as_groups();
	assert(false == groups.empty());
	groups[0] = shape0;
	if (cdim < ashape.n_rank())
	{
		if (groups.size() == 1)
		{
			groups.push_back(shape1);
		}
		else
		{
			groups[1] = shape1;
		}
	}

	return Shape(groups);
}

static EnumMap<OPCODE,Shaper> shapers =
{
	{TRANSPOSE, transpose_shaper},
	{ADD, elem_shaper},
	{MUL, elem_shaper},
	{MATMUL, matmul_shaper},
};

Shaper get_shaper (OPCODE opcode)
{
	auto it = shapers.find(opcode);
	if (shapers.end() == it)
	{
		handle_error("failed to retrieve shaper",
			ErrArg<std::string>("opcode", opname(opcode)));
	}
	return it->second;
}

#endif
