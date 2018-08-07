#include <cassert>

#include "sand/preop.hpp"
#include "util/error.hpp"
#include "util/mapper.hpp"

#ifdef SAND_PREOP_HPP

Meta ElemPreOperator::operator () (std::vector<Meta> args)
{
	if (args.size() == 0)
	{
		handle_error("creating elementary shape from no args",
			ErrArg<size_t>{"num_args", args.size()});
	}

	Meta out = args.front();
	uint8_t outrank = out.shape_.n_rank();
	for (auto it = args.begin() + 1, et = args.end(); it != et; ++it)
	{
		Meta arg = *it;
		uint8_t rank = arg.shape_.n_rank();
		if (false == out.compatible(arg))
		{
			handle_error("incompatible elem arg",
				ErrArg<std::string>{"first", out.to_string()},
				ErrArg<std::string>{"cur", arg.to_string()});
		}
		if (rank > outrank)
		{
			out.shape_ = arg.shape_;
			outrank = rank;
		}
	}
	return out;
}

MetaEncoder ElemPreOperator::encode (void) const
{
	return MetaEncoder(scode_);
}

TransPreOperator::TransPreOperator (SortedArr<uint8_t,4> groups) : groups_(groups)
{
	if (groups[3] > rank_cap)
	{
		handle_error("transposing on out or rank groups",
			ErrArg<unsigned>("top_rank", groups[3]));
	}
}

Meta TransPreOperator::operator () (std::vector<Meta> args)
{
	if (args.size() != 1)
	{
		handle_error("creating tranpose shape from multiple args",
			ErrArg<size_t>{"num_args", args.size()});
	}

	Meta& arg = args[0];
	auto it = arg.shape_.begin();
	std::vector<DimT> olist(it, it + groups_[0]);
	olist.insert(olist.end(), it + groups_[2], it + groups_[3]);
	olist.insert(olist.end(), it + groups_[1], it + groups_[2]);
	olist.insert(olist.end(), it + groups_[0], it + groups_[1]);
	olist.insert(olist.end(), it + groups_[3], arg.shape_.end());

	// std::vector<Shape> olist = {Shape(std::vector<DimT>(it, it + groups_[0]))};
	// olist.push_back(Shape(std::vector<DimT>(it + groups_[2], it + groups_[3])));
	// olist.push_back(Shape(std::vector<DimT>(it + groups_[1], it + groups_[2])));
	// olist.push_back(Shape(std::vector<DimT>(it + groups_[0], it + groups_[1])));
	// olist.push_back(Shape(std::vector<DimT>(it + groups_[3], arg.shape_.end())));

	return Meta{Shape(olist), arg.type_};
}

MetaEncoder TransPreOperator::encode (void) const
{
	MetaEncoder out(scode_);
	std::memcpy(out.data_, &groups_[0], 4);
	return out;
}

MatPreOperator::MatPreOperator (TwoGroups groups0, TwoGroups groups1) :
	groups0_(groups0), groups1_(groups1)
{
	if (groups0[1] > rank_cap || groups1[1] > rank_cap)
	{
		handle_error("matmul on out or rank groups",
			ErrArg<unsigned>("top_rank0", groups0[1]),
			ErrArg<unsigned>("top_rank1", groups1[1]));
	}
	uint8_t group0_rank = groups0[0];
	uint8_t group1_rank = groups1[1] - groups1[0];
	if (group0_rank != group1_rank)
	{
		handle_error("incompatible rank of common dimension",
			ErrArg<unsigned>("rank0", group0_rank),
			ErrArg<unsigned>("rank1", group1_rank));
	}
}

Meta MatPreOperator::operator () (std::vector<Meta> args)
{
	if (args.size() != 2)
	{
		handle_error("creating matmul shape not from 2 args",
			ErrArg<size_t>{"num_args", args.size()});
	}

	Meta a = args.front();
	Meta b = args.back();
	if (a.type_ != b.type_)
	{
		handle_error("incompatible types in matmul",
			ErrArg<std::string>{"a", a.to_string()},
			ErrArg<std::string>{"b", b.to_string()});
	}
	auto it0 = a.shape_.begin();
	auto it1 = b.shape_.begin();
	if (false == std::equal(it0,
		it0 + groups0_[0], it1 + groups1_[0]))
	{
		handle_error("incompatible common dimension in matmul",
			ErrArg<std::string>{"a", a.to_string()},
			ErrArg<std::string>{"b", b.to_string()});
	}

	uint8_t rank = rank_cap - std::max(groups0_[1], groups1_[1]);
	if (false == std::equal(it0 + groups0_[1],
		it0 + rank, it1 + groups1_[1]))
	{
		handle_error("incompatible dimensions beyond first 2 groups in matmul",
			ErrArg<std::string>{"a", a.to_string()},
			ErrArg<std::string>{"b", b.to_string()});
	}

	// std::vector<DimT> olist(it1, it1 + groups1_[0]);
	// olist.insert(olist.end(), it0 + groups0_[0], it0 + a.shape_.n_rank());

	std::vector<Shape> olist = {Shape(std::vector<DimT>(it1, it1 + groups1_[0]))};
	uint8_t arank = a.shape_.n_rank();
	if (groups0_[0] < arank)
	{
		olist.push_back(Shape(std::vector<DimT>(it0 + groups0_[0], it0 + groups0_[1])));
	}
	if (groups0_[1] < arank)
	{
		olist.push_back(Shape(std::vector<DimT>(it0 + groups0_[1], it0 + arank)));
	}

	return Meta{Shape(olist), a.type_};
}

MetaEncoder MatPreOperator::encode (void) const
{
	MetaEncoder out(scode_);
	std::memcpy(out.data_, &groups0_[0], 2);
	std::memcpy(out.data_ + 2, &groups1_[0], 2);
	return out;
}

std::shared_ptr<iPreOperator> decode_meta (std::string msg)
{
	switch ((SCODE) msg[0])
	{
		case ELEM:
			return std::make_shared<ElemPreOperator>();
		case TSHAPE:
			return std::shared_ptr<TransPreOperator>(
				new TransPreOperator({(uint8_t) msg[1], (uint8_t) msg[2],
				(uint8_t) msg[3], (uint8_t) msg[4]}));
		case MATSHAPE:
			return std::shared_ptr<MatPreOperator>(
				new MatPreOperator({(uint8_t) msg[1], (uint8_t) msg[2]},
				{(uint8_t) msg[3], (uint8_t) msg[4]}));
	}
	return nullptr;
}

#endif
