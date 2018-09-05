#include <cstring>

#include "ade/fwder.hpp"

#include "util/error.hpp"

#ifdef ADE_FWDER_HPP

namespace ade
{

#define BIJECT(CODE)\
template <> Shape forwarder<CODE> (std::vector<Tensorptr> tens)\
{ return bijection(tens, opname(CODE)); }

#define SCALAR(CODE)\
template <> Shape forwarder<CODE> (std::vector<Tensorptr> tens)\
{ if (1 != tens.size()) {\
	util::handle_error("wrong number of arguments (expected 1)",\
		util::ErrArg<size_t>("got", tens.size()));\
} return Shape(); }

static Shape bijection (std::vector<Tensorptr> args, std::string op)
{
	if (args.size() == 0)
	{
		util::handle_error(op + " error: creating elementary shape from no args",
			util::ErrArg<size_t>{"num_args", args.size()});
	}

	Shape& outshape = args[0]->shape_;
	uint8_t outrank = outshape.n_rank();
	for (auto it = args.begin() + 1, et = args.end(); it != et; ++it)
	{
		Shape& shape = (*it)->shape_;
		uint8_t rank = shape.n_rank();
		if (false == shape.compatible_before(outshape,
			std::min(outrank, rank)))
		{
			util::handle_error(op + " error: incompatible shapes",
				util::ErrArg<std::vector<DimT>>{"outshape", outshape.as_list()},
				util::ErrArg<std::vector<DimT>>{"shape", shape.as_list()});
		}
		if (rank > outrank)
		{
			outshape = shape;
			outrank = rank;
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

BIJECT(BINO)
BIJECT(UNIF)
BIJECT(NORM)

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
		util::handle_error("wrong number of arguments (expected 2)",
			util::ErrArg<size_t>("got", tens.size()));
	}
	if (agroup_idx == 0)
	{
		util::handle_error("agroup_idx == 0");
	}
	if (bgroup_idx == 0)
	{
		util::handle_error("bgroup_idx == 0");
	}
	Shape& ashape = tens[0]->shape_;
	Shape& bshape = tens[1]->shape_;
	uint8_t arank = ashape.n_rank();
	uint8_t brank = bshape.n_rank();
	if (agroup_idx > arank)
	{
		util::handle_error("agroup_idx > rank",
			util::ErrArg<size_t>("agroup_idx", agroup_idx),
			util::ErrArg<size_t>("rank", arank));
	}
	if (bgroup_idx > brank)
	{
		util::handle_error("bgroup_idx > rank",
			util::ErrArg<size_t>("bgroup_idx", bgroup_idx),
			util::ErrArg<size_t>("rank", brank));
	}
	auto ait = ashape.begin();
	auto bit = bshape.begin();
	if (false == std::equal(ait, ait + agroup_idx, bit + bgroup_idx) ||
		agroup_idx != (bshape.n_rank() - bgroup_idx))
	{
		util::handle_error("incompatible common dimension in matmul",
			util::ErrArg<std::vector<DimT>>{"a", ashape.as_list()},
			util::ErrArg<std::vector<DimT>>{"b", bshape.as_list()});
	}

	std::vector<DimT> outlist(bit, bit + bgroup_idx);
	outlist.insert(outlist.end(), ait + agroup_idx, ashape.end());
	return Shape(outlist);
}

template <>
Shape forwarder<PERMUTE,std::vector<uint8_t>> (std::vector<Tensorptr> tens, std::vector<uint8_t> dims)
{
	if (1 != tens.size())
	{
		util::handle_error("wrong number of arguments (expected 1)",
			util::ErrArg<size_t>("got", tens.size()));
	}
	Shape& shape = tens[0]->shape_;
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
		util::handle_error("wrong number of arguments (expected 1)",
			util::ErrArg<size_t>("got", tens.size()));
	}
	if (0 == ext.size())
	{
		util::handle_error("empty extension to shape");
	}
	if ((tens[0]->shape_.n_rank() + ext.size()) >= rank_cap)
	{
		util::handle_error("failed attempt to extend dimension to beyond rank_cap",
			util::ErrArg<std::vector<DimT>>("shape", tens[0]->shape_.as_list()),
			util::ErrArg<std::vector<DimT>>("ext", ext));
	}
	std::vector<DimT> outlist = tens[0]->shape_.as_list();
	outlist.insert(outlist.end(), ext.begin(), ext.end());
	return Shape(outlist);
}

template <>
Shape forwarder<RESHAPE,std::vector<DimT>> (
	std::vector<Tensorptr> tens, std::vector<DimT> outlist)
{
	if (1 != tens.size())
	{
		util::handle_error("wrong number of arguments (expected 1)",
			util::ErrArg<size_t>("got", tens.size()));
	}
	Shape outshape(outlist);
	NElemT nin = tens[0]->shape_.n_elems();
	NElemT nout = outshape.n_elems();
	if (1 < nin && nin != nout)
	{
		util::handle_error("input can't be easily expanded to output",
			util::ErrArg<NElemT>("nin", nin),
			util::ErrArg<NElemT>("nout", nout));
	}
	return outshape;
}

#undef BIJECT

#undef SCALAR

}

#endif
