#include <cstring>

#include "ade/log.hpp"

#include "ade/fwder.hpp"

#ifdef ADE_FWDER_HPP

namespace ade
{

#define BIJECT(CODE)\
template <> Shape forwarder<CODE> (std::vector<Tensorptr> tens)\
{ return bijection(tens, opname(CODE)); }

#define SCALAR(CODE)\
template <> Shape forwarder<CODE> (std::vector<Tensorptr> tens)\
{ if (1 != tens.size()) {\
	fatalf("cannot %s for non-single argument(s): "\
		"using %d arguments", #CODE, tens.size());\
} return Shape(); }

static Shape bijection (std::vector<Tensorptr> args, std::string op)
{
	if (args.size() == 0)
	{
		fatalf("cannot %s with no arguments", op.c_str());
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
			fatalf("cannot %s with incompatible shapes %s and %s", op.c_str(),
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

BIJECT(ABS)
BIJECT(NEG)
BIJECT(NOT)
BIJECT(SIN)
BIJECT(COS)
BIJECT(TAN)
BIJECT(EXP)
BIJECT(LOG)
BIJECT(SQRT)
BIJECT(ROUND)
BIJECT(FLIP)

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

SCALAR(N_ELEMS)
SCALAR(N_DIMS)
SCALAR(ARGMAX)
SCALAR(RMAX)
SCALAR(RSUM)

template <>
Shape forwarder<MATMUL> (std::vector<Tensorptr> tens)
{
	return forwarder<MATMUL,uint8_t,uint8_t>(tens,1,1);
}

template <>
Shape forwarder<MATMUL,uint8_t,uint8_t> (std::vector<Tensorptr> tens,
	uint8_t agroup_idx, uint8_t bgroup_idx)
{
	if (2 != tens.size())
	{
		fatalf("cannot MATMUL without 2 arguments: "
			"using %d arguments", tens.size());
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
		fatalf("incompatible common dimensions in matmuling %s, %s",
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
			"using %d arguments", tens.size());
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
			"using %d arguments", tens.size());
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

template <>
Shape forwarder<RESHAPE,std::vector<DimT>> (
	std::vector<Tensorptr> tens, std::vector<DimT> outlist)
{
	if (1 != tens.size())
	{
		fatalf("cannot RESHAPE non-single argument(s): "
			"using %d arguments", tens.size());
	}
	Shape inshape = tens[0]->shape();
	Shape outshape(outlist);
	NElemT nin = inshape.n_elems();
	NElemT nout = outshape.n_elems();
	if (1 < nin && nin != nout)
	{
		fatalf("cannot RESHAPE non-scalar shape %s to shape %s",
			inshape.to_string().c_str(), outshape.to_string().c_str());
	}
	return outshape;
}

#undef BIJECT

#undef SCALAR

}

#endif
