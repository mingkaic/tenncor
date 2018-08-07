#include <functional>

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

struct MatPreOperator final : public iPreOperator
{
	using TwoGroups = SortedArr<uint8_t,2>;

	static const SCODE scode_ = MATSHAPE;

	MatPreOperator (TwoGroups groups0, TwoGroups groups1);

	Meta operator () (std::vector<Meta> args) override;

	MetaEncoder encode (void) const override;

private:
	TwoGroups groups0_;
	TwoGroups groups1_;
};

std::shared_ptr<iPreOperator> decode_meta (std::string msg);

#endif /* SAND_PREOP_HPP */
