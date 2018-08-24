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

	ElemPreOperator (void);

	ElemPreOperator (std::vector<uint8_t> save);

	Meta operator () (std::vector<Meta> args) override;

	MetaEncoder encode (void) const override;

private:
	MetaEncoder::MetaData save_;
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

	TypecastPreOperator (DTYPE outtype, DTYPE intype);

	Meta operator () (std::vector<Meta> args) override;

	MetaEncoder encode (void) const override;

private:
	DTYPE outtype_;
	DTYPE intype_;
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
			Shape(std::vector<DimT>(it + group_[0],
				it + std::min(group_[1], srcshape.n_rank())))
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

	Meta operator () (std::vector<Meta> args) override;

	MetaEncoder encode (void) const override;
};

struct NDimsPreOperator final : public iPreOperator
{
	static const SCODE scode_ = NDIMSPRE;

	NDimsPreOperator (void);

	NDimsPreOperator (uint8_t dim);

	Meta operator () (std::vector<Meta> args) override;

	MetaEncoder encode (void) const override;

private:
	uint8_t dim_;
};

struct BinomPreOperator final : public iPreOperator
{
	static const SCODE scode_ = BINOPRE;

	Meta operator () (std::vector<Meta> args) override;

	MetaEncoder encode (void) const override;
};

struct ReducePreOperator final : public iPreOperator
{
	static const SCODE scode_ = REDUCEPRE;

	ReducePreOperator (void);

	ReducePreOperator (uint8_t dim);

	Meta operator () (std::vector<Meta> args) override;

	MetaEncoder encode (void) const override;

private:
	uint8_t dim_;
};

std::shared_ptr<iPreOperator> decode_meta (std::string msg);

#endif /* SAND_PREOP_HPP */
