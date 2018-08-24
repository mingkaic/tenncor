#include <cstring>

#include "sand/preop.hpp"
#include "util/error.hpp"
#include "util/mapper.hpp"

#ifdef SAND_PREOP_HPP

ElemPreOperator::ElemPreOperator (void)
{
	std::memset(save_, 0, MetaEncoder::NHash);
}

ElemPreOperator::ElemPreOperator (std::vector<uint8_t> save) :
	ElemPreOperator()
{
	uint8_t n = std::min(save.size(), (size_t) MetaEncoder::NHash);
	for (uint8_t i = 0; i < n; ++i)
	{
		save_[i] = save[i];
	}
}

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
		if (out.type_ != arg.type_ ||
			false == arg.shape_.compatible_before(out.shape_,
			std::min(outrank, rank)))
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
	MetaEncoder out(scode_);
	std::memcpy(out.data_, save_, MetaEncoder::NHash);
	return out;
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
	uint8_t arank = arg.shape_.n_rank();
	// std::vector<DimT> olist(it, it + groups_[0]);
	// olist.insert(olist.end(), it + groups_[2], it + groups_[3]);
	// olist.insert(olist.end(), it + groups_[1], it + groups_[2]);
	// olist.insert(olist.end(), it + groups_[0], it + groups_[1]);
	// if (groups_[3] < arank)
	// {
	// 	olist.insert(olist.end(), it + groups_[3], it + arank);
	// }

	std::vector<Shape> olist = {Shape(std::vector<DimT>(it, it + groups_[0]))};
	olist.push_back(Shape(std::vector<DimT>(it + groups_[2], it + groups_[3])));
	olist.push_back(Shape(std::vector<DimT>(it + groups_[1], it + groups_[2])));
	olist.push_back(Shape(std::vector<DimT>(it + groups_[0], it + groups_[1])));
	if (groups_[3] < arank)
	{
		olist.push_back(Shape(std::vector<DimT>(it + groups_[3], it + arank)));
	}

	return Meta{Shape(olist), arg.type_};
}

MetaEncoder TransPreOperator::encode (void) const
{
	MetaEncoder out(scode_);
	std::memcpy(out.data_, &groups_[0], 4);
	return out;
}

MatPreOperator::MatPreOperator (Range groups0, Range groups1) :
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

	if (false == std::equal(it0 + groups0_[1],
		it0 + std::max(groups0_[1], a.shape_.n_rank()), it1 + groups1_[1]))
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

TypecastPreOperator::TypecastPreOperator (DTYPE outtype, DTYPE intype) :
	outtype_(outtype), intype_(intype) {}

Meta TypecastPreOperator::operator () (std::vector<Meta> args)
{
	if (args.size() != 1)
	{
		handle_error("cannot aggregate multiple arguments from cast",
			ErrArg<size_t>{"num_args", args.size()});
	}
	if (args[0].type_ != intype_)
	{
		handle_error("unexpected conversion",
			ErrArg<std::string>("expected_intype", name_type(intype_)),
			ErrArg<std::string>("got_intype", name_type(args[0].type_)));
	}
	return Meta{args[0].shape_, outtype_};
}

MetaEncoder TypecastPreOperator::encode (void) const
{
	MetaEncoder out(scode_);
	out.data_[0] = (uint8_t) outtype_;
	out.data_[1] = (uint8_t) intype_;
	return out;
}

Meta NElemsPreOperator::operator () (std::vector<Meta> args)
{
	if (args.size() != 1)
	{
		handle_error("cannot aggregate multiple arguments from nelems",
			ErrArg<size_t>{"num_args", args.size()});
	}
	return Meta{Shape({1}), UINT32};
}

MetaEncoder NElemsPreOperator::encode (void) const
{
	return MetaEncoder(scode_);
}

NDimsPreOperator::NDimsPreOperator (void) : dim_(rank_cap) {}

NDimsPreOperator::NDimsPreOperator (uint8_t dim) : dim_(dim)
{
	if (dim >= rank_cap)
	{
		handle_error("ndims dimension out of bounds",
			ErrArg<size_t>{"dim", dim});
	}
}

Meta NDimsPreOperator::operator () (std::vector<Meta> args)
{
	if (args.size() != 1)
	{
		handle_error("cannot aggregate multiple arguments with ndims",
			ErrArg<size_t>{"num_args", args.size()});
	}
	DimT d = 1;
	if (dim_ < rank_cap)
	{
		d = args[0].shape_.at(dim_);
	}
	return Meta{Shape({d}), UINT8};
}

MetaEncoder NDimsPreOperator::encode (void) const
{
	MetaEncoder out(scode_);
	out.data_[0] = dim_;
	return out;
}

Meta BinomPreOperator::operator () (std::vector<Meta> args)
{
	if (args.size() != 2)
	{
		handle_error("binomial distribution takes 2 arguments",
			ErrArg<size_t>{"num_args", args.size()});
	}
	if (DOUBLE != args[1].type_)
	{
		handle_error(
			"second argument of binom (probability) must be type double",
			ErrArg<std::string>("got_type", name_type(args[1].type_)));
	}
	uint8_t rank0 = args[0].shape_.n_rank();
	uint8_t rank1 = args[1].shape_.n_rank();
	if (false == args[0].shape_.compatible_before(args[1].shape_,
		std::min(rank0, rank1)))
	{
		handle_error("incompatible elem arg",
			ErrArg<std::string>{"a", args[0].to_string()},
			ErrArg<std::string>{"b", args[1].to_string()});
	}
	if (rank1 > rank0)
	{
		return Meta{args[1].shape_, args[0].type_};
	}
	return args[0];
}

MetaEncoder BinomPreOperator::encode (void) const
{
	return MetaEncoder(scode_);
}

ReducePreOperator::ReducePreOperator (void) : dim_(rank_cap) {}

ReducePreOperator::ReducePreOperator (uint8_t dim) : dim_(dim)
{
	if (dim >= rank_cap)
	{
		handle_error("ndims dimension out of bounds",
			ErrArg<size_t>{"dim", dim});
	}
}

Meta ReducePreOperator::operator () (std::vector<Meta> args)
{
	if (args.size() != 1)
	{
		handle_error("cannot aggregate multiple arguments with reduction",
			ErrArg<size_t>{"num_args", args.size()});
	}
	if (dim_ < rank_cap)
	{
		auto slist = args[0].shape_.as_list();
		slist.erase(slist.begin() + dim_);
		return Meta{Shape(slist), args[0].type_};
	}
	return Meta{Shape({1}), args[0].type_};
}

MetaEncoder ReducePreOperator::encode (void) const
{
	MetaEncoder out(scode_);
	out.data_[0] = dim_;
	return out;
}

std::shared_ptr<iPreOperator> decode_meta (std::string msg)
{
	switch ((SCODE) msg[0])
	{
		case ELEM:
			return std::shared_ptr<ElemPreOperator>(
				new ElemPreOperator({
					(uint8_t) msg[1], (uint8_t) msg[2],
					(uint8_t) msg[3], (uint8_t) msg[4]}));
		case TSHAPE:
			return std::shared_ptr<TransPreOperator>(
				new TransPreOperator({
					(uint8_t) msg[1], (uint8_t) msg[2],
					(uint8_t) msg[3], (uint8_t) msg[4]}));
		case MATSHAPE:
			return std::shared_ptr<MatPreOperator>(
				new MatPreOperator(
					{(uint8_t) msg[1], (uint8_t) msg[2]},
					{(uint8_t) msg[3], (uint8_t) msg[4]}));
		case TCAST:
			return std::shared_ptr<TypecastPreOperator>(
				new TypecastPreOperator((DTYPE) msg[0], (DTYPE) msg[1]));
		case GROUP:
			return std::shared_ptr<GroupPreOperator>(
				new GroupPreOperator({(uint8_t) msg[1], (uint8_t) msg[2]}));
		case NELEMSPRE:
			return std::make_shared<NElemsPreOperator>();
		case NDIMSPRE:
		{
			if (msg[1] > 0)
			{
				return std::shared_ptr<NDimsPreOperator>(
					new NDimsPreOperator(msg[1] - 1));
			}
			return std::make_shared<NDimsPreOperator>();
		}
		case REDUCEPRE:
		{
			if (msg[1] > 0)
			{
				return std::shared_ptr<ReducePreOperator>(
					new ReducePreOperator(msg[1] - 1));
			}
			return std::make_shared<ReducePreOperator>();
		}
		case BINOPRE:
			return std::make_shared<BinomPreOperator>();
	}
	return nullptr;
}

#endif
