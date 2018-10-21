#include <cstring>

#include "ade/log.hpp"

#include "ade/fwder.hpp"

#ifdef ADE_FWDER_HPP

namespace ade
{

#define UNARY(CODE)template <> Shape forwarder<CODE> (\
std::vector<Tensorptr> tens) { return unary(tens, #CODE); }

#define BIJECT(CODE)template <> Shape forwarder<CODE> (\
std::vector<Tensorptr> tens) { return bijection(tens, #CODE); }

#define REDUCE(CODE)template <> Shape forwarder<CODE> (\
std::vector<Tensorptr> tens, uint8_t dim) {\
return reduction(tens, dim, #CODE); }

static Shape unary (std::vector<Tensorptr>& args, const char* op)
{
	if (1 != args.size())
	{
		fatalf("cannot %s for non-single argument(s): using %d argument(s)",
			op, args.size());
	}

	return args[0]->shape();
}

static Shape bijection (std::vector<Tensorptr>& args, const char* op)
{
	if (0 == args.size())
	{
		fatalf("cannot %s with no arguments", op);
	}

	Shape outshape = args[0]->shape();
	uint8_t outrank = outshape.n_rank();
	NElemT outn = outshape.n_elems();
	for (auto it = args.begin() + 1, et = args.end(); it != et; ++it)
	{
		const Shape& shape = (*it)->shape();
		uint8_t rank = shape.n_rank();
		NElemT nelems = shape.n_elems();
		if (false == shape.compatible_before(outshape,
			std::min(outrank, rank)) && std::min(nelems, outn) > 1)
		{
			fatalf("cannot %s with incompatible shapes %s and %s", op,
				outshape.to_string().c_str(), shape.to_string().c_str());
		}
		if (nelems > outn)
		{
			outshape = shape;
			outrank = rank;
			outn = nelems;
		}
	}
	return outshape;
}

static Shape reduction (std::vector<Tensorptr>& args, uint8_t dim, const char* op)
{
	if (1 != args.size())
	{
		fatalf("cannot %s for non-single argument(s): using %d argument(s)",
			op, args.size());
	}
	if (dim == 0)
	{
		warn("reducing coordinates [:0] ... created useless node");
	}
	const Shape& shape = args.front()->shape();
	auto it = shape.begin();
	uint8_t rank = shape.n_rank();
	return Shape(std::vector<ade::DimT>(it + std::min(rank, dim), it + rank));
}

UNARY(ABS)
UNARY(NEG)
UNARY(NOT)
UNARY(SIN)
UNARY(COS)
UNARY(TAN)
UNARY(EXP)
UNARY(LOG)
UNARY(SQRT)
UNARY(ROUND)
UNARY(FLIP)

BIJECT(POW)
BIJECT(ADD)
BIJECT(SUB)
BIJECT(MUL)
BIJECT(DIV)
BIJECT(EQ)
BIJECT(NE)
BIJECT(LT)
BIJECT(GT)
BIJECT(MIN)
BIJECT(MAX)

BIJECT(RAND_BINO)
BIJECT(RAND_UNIF)
BIJECT(RAND_NORM)

REDUCE(ARGMAX)
REDUCE(RMAX)
REDUCE(RSUM)

template <>
Shape forwarder<MATMUL,uint8_t,uint8_t> (std::vector<Tensorptr> tens,
	uint8_t agroup_idx, uint8_t bgroup_idx)
{
	if (2 != tens.size())
	{
		fatalf("cannot MATMUL without 2 arguments: "
			"using %d argument(s)", tens.size());
	}
	if (agroup_idx == 0)
	{
		fatal("agroup_idx == 0");
	}
	if (bgroup_idx == 0)
	{
		fatal("bgroup_idx == 0");
	}
	const Shape& ashape = tens[0]->shape();
	const Shape& bshape = tens[1]->shape();
	uint8_t arank = ashape.n_rank();
	uint8_t brank = bshape.n_rank();
	if (agroup_idx > arank)
	{
		fatalf("agroup_idx %d > rank %d",
			agroup_idx, arank);
	}
	if (bgroup_idx > brank)
	{
		fatalf("bgroup_idx %d > rank %d",
			bgroup_idx, brank);
	}
	const auto ait = ashape.begin();
	const auto bit = bshape.begin();
	if (false == std::equal(ait, ait + agroup_idx, bit + bgroup_idx) ||
		agroup_idx != (bshape.n_rank() - bgroup_idx))
	{
		fatalf("incompatible common dimensions when matmuling shapes %s, %s",
			ashape.to_string().c_str(), bshape.to_string().c_str());
	}

	std::vector<DimT> outlist(bit, bit + bgroup_idx);
	outlist.insert(outlist.end(), ait + agroup_idx, ait + arank);
	return Shape(outlist);
}

template <>
Shape forwarder<PERMUTE,std::vector<uint8_t>> (
	std::vector<Tensorptr> tens, std::vector<uint8_t> dims)
{
	if (1 != tens.size())
	{
		fatalf("cannot PERMUTE non-single argument(s): "
			"using %d argument(s)", tens.size());
	}
	const Shape& shape = tens[0]->shape();
	bool visited[rank_cap];
	std::memset(visited, false, rank_cap);
	std::vector<DimT> outlist(dims.size(), 0);
	for (size_t i = 0, n = dims.size(); i < n; ++i)
	{
		outlist[i] = shape.at(dims[i]);
		visited[dims[i]] = true;
	}
	for (size_t i = 0, n = shape.n_rank(); i < n; ++i)
	{
		if (false == visited[i])
		{
			outlist.push_back(shape.at(i));
		}
	}
	return Shape(outlist);
}

template <>
Shape forwarder<EXTEND,std::vector<DimT>> (
	std::vector<Tensorptr> tens, std::vector<DimT> ext)
{
	if (1 != tens.size())
	{
		fatalf("cannot EXTEND non-single argument(s): "
			"using %d argument(s)", tens.size());
	}
	ade::Shape shape = tens[0]->shape();
	if (0 == ext.size())
	{
		warn("EXTENDing with empty vector... created useless node");
	}
	else
	{
		if ((shape.n_rank() + ext.size()) > rank_cap)
		{
			fatalf("cannot EXTEND dimension beyond rank_cap using "
				"vector %s on shape %s",
				to_string(ext).c_str(), shape.to_string().c_str());
		}
		std::vector<DimT> outlist = shape.as_list();
		outlist.insert(outlist.end(), ext.begin(), ext.end());
		shape = Shape(outlist);
	}
	return shape;
}

#undef UNARY

#undef BIJECT

#undef REDUCE

}

#endif
