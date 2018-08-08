#include <functional>
#include <memory>

#include "util/sorted_arr.hpp"

#include "sand/opcode.hpp"
#include "sand/meta.hpp"

#ifndef SAND_PREOP_HPP
#define SAND_PREOP_HPP

struct iPreOperator
{
	virtual ~iPreOperator (void) = default;

	virtual Meta operator () (std::vector<Meta> args) = 0;

	virtual MetaEncoder encode (void) const = 0;
};

struct ElemPreOperator final : public iPreOperator
{
	static const SCODE scode_ = ELEM;

	Meta operator () (std::vector<Meta> args) override;

	MetaEncoder encode (void) const override;
};

struct TransPreOperator final : public iPreOperator
{
	static const SCODE scode_ = TSHAPE;

	TransPreOperator (SortedArr<uint8_t,4> groups);

	Meta operator () (std::vector<Meta> args) override;

	MetaEncoder encode (void) const override;

private:
	SortedArr<uint8_t,4> groups_;
};

using Range = SortedArr<uint8_t,2>;

struct MatPreOperator final : public iPreOperator
{
	static const SCODE scode_ = MATSHAPE;

	MatPreOperator (Range groups0, Range groups1);

	Meta operator () (std::vector<Meta> args) override;

	MetaEncoder encode (void) const override;

private:
	// range of second group of ith argument,
	// first group between 0 and beginning of second
	Range groups0_;
	Range groups1_;
};

struct TypecastPreOperator final : public iPreOperator
{
	static const SCODE scode_ = TCAST;

	TypecastPreOperator (DTYPE type, DTYPE arg) : type_(type), arg_(arg) {}

	Meta operator () (std::vector<Meta> args) override
	{
		if (args.size() != 1)
		{
			handle_error("cannot aggregate multiple arguments from cast",
				ErrArg<size_t>{"num_args", args.size()});
		}
		return Meta{args[0].shape_, type_};
	}

	MetaEncoder encode (void) const override
	{
		MetaEncoder out(scode_);
		out.data_[0] = (uint8_t) type_;
		out.data_[1] = (uint8_t) arg_;
		return out;
	}

private:
	DTYPE type_;
	DTYPE arg_;
};

struct FlipPreOperator final : public iPreOperator
{
	static const SCODE scode_ = FLIPSHAPE;

	FlipPreOperator (uint8_t dim) : dim_(dim) {}

	Meta operator () (std::vector<Meta> args) override
	{
		if (args.size() != 1)
		{
			handle_error("cannot aggregate multiple arguments from flip",
				ErrArg<size_t>{"num_args", args.size()});
		}
		return args[0];
	}

	MetaEncoder encode (void) const override
	{
		MetaEncoder out(scode_);
		out.data_[0] = dim_;
		return out;
	}

private:
	uint8_t dim_;
};

struct GroupPreOperator final : public iPreOperator
{
	static const SCODE scode_ = GROUP;

	GroupPreOperator (Range group) : group_(group) {}

	Meta operator () (std::vector<Meta> args) override
	{
		if (args.size() != 1)
		{
			handle_error("cannot aggregate multiple arguments from group",
				ErrArg<size_t>{"num_args", args.size()});
		}
		Shape& srcshape = args[0].shape_;
		auto it = srcshape.begin();
		std::vector<Shape> olist = {
			Shape(std::vector<DimT>(it, it + group_[0])),
			Shape(std::vector<DimT>(it + group_[0], it + group_[1]))
		};
		uint8_t srank = srcshape.n_rank();
		if (group_[1] < srank)
		{
			olist.push_back(
				Shape(std::vector<DimT>(it + group_[1], it + srank)));
		}
		return Meta{Shape(olist), args[0].type_};
	}

	MetaEncoder encode (void) const override
	{
		MetaEncoder out(scode_);
		std::memcpy(out.data_, &group_[0], 2);
		return out;
	}

private:
	Range group_;
};

struct NElemsPreOperator final : public iPreOperator
{
	static const SCODE scode_ = NELEMSPRE;

	Meta operator () (std::vector<Meta> args) override
	{
		if (args.size() != 1)
		{
			handle_error("cannot aggregate multiple arguments from nelems",
				ErrArg<size_t>{"num_args", args.size()});
		}
		return Meta{Shape({1}), UINT32};
	}

	MetaEncoder encode (void) const override
	{
		return MetaEncoder(scode_);
	}
};

struct NDimsPreOperator final : public iPreOperator
{
	static const SCODE scode_ = NDIMSPRE;

	NDimsPreOperator (void) : dim_(0) {}

	NDimsPreOperator (uint8_t dim) : dim_(dim + 1) {}

	Meta operator () (std::vector<Meta> args) override
	{
		if (args.size() != 1)
		{
			handle_error("cannot aggregate multiple arguments with ndims",
				ErrArg<size_t>{"num_args", args.size()});
		}
		DimT d = 1;
		if (dim_ > 0)
		{
			d = args[0].shape_.at(dim_ - 1);
		}
		return Meta{Shape({d}), UINT8};
	}

	MetaEncoder encode (void) const override
	{
		MetaEncoder out(scode_);
		out.data_[0] = dim_;
		return out;
	}

private:
	uint8_t dim_;
};

struct BinomPreOperator final : public iPreOperator
{
	static const SCODE scode_ = BINOPRE;

	Meta operator () (std::vector<Meta> args) override
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

	MetaEncoder encode (void) const override
	{
		return MetaEncoder(scode_);
	}
};

struct ReducePreOperator final : public iPreOperator
{
	static const SCODE scode_ = REDUCEPRE;

	ReducePreOperator (void) : dim_(0) {}

	ReducePreOperator (uint8_t dim) : dim_(dim + 1) {}

	Meta operator () (std::vector<Meta> args) override
	{
		if (args.size() != 1)
		{
			handle_error("cannot aggregate multiple arguments with reduction",
				ErrArg<size_t>{"num_args", args.size()});
		}
		if (dim_ > 0)
		{
			auto slist = args[0].shape_.as_list();
			slist.erase(slist.begin() + dim_);
			return Meta{Shape(slist), args[0].type_};
		}
		return Meta{Shape({1}), args[0].type_};
	}

	MetaEncoder encode (void) const override
	{
		MetaEncoder out(scode_);
		out.data_[0] = dim_;
		return out;
	}

private:
	uint8_t dim_;
};

std::shared_ptr<iPreOperator> decode_meta (std::string msg);

#endif /* SAND_PREOP_HPP */
