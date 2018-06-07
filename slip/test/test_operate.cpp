#ifndef DISABLE_SLIP_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "clay/memory.hpp"

#include "slip/registry.hpp"
#include "slip/error.hpp"


#ifndef DISABLE_OPERATE_IO_TEST


using namespace testutil;


class OPERATE_IO : public fuzz_test {};


void set_op (fuzz_test* fuzzer, mold::iOperatePtrT& op, slip::OPCODE opcode)
{
	switch (opcode)
	{
		case slip::ABS:
		case slip::NEG:
		case slip::NOT:
		case slip::SIN:
		case slip::COS:
		case slip::TAN:
		case slip::EXP:
		case slip::LOG:
		case slip::SQRT:
		case slip::ROUND:
		case slip::TRANSPOSE:
		case slip::UARGMAX:
		case slip::URMAX:
		case slip::URSUM:
		case slip::N_ELEMS:
		{
			clay::Shape shape = random_def_shape(fuzzer);
			clay::DTYPE dtype = (clay::DTYPE)
				fuzzer->get_int(1, "dtype", {1, clay::_SENTINEL - 1})[0];
			size_t nbytes = shape.n_elems() * clay::type_size(dtype);
			std::shared_ptr<char> data = clay::make_char(nbytes);
			op->set_args({clay::State(data, shape, dtype)});
		}
		break;
		case slip::CAST:
		case slip::POW:
		case slip::ADD:
		case slip::SUB:
		case slip::MUL:
		case slip::DIV:
		case slip::EQ:
		case slip::NE:
		case slip::GT:
		case slip::LT:
		case slip::UNIF:
		{
			clay::Shape shape = random_def_shape(fuzzer);
			clay::DTYPE dtype = (clay::DTYPE)
				fuzzer->get_int(1, "dtype", {1, clay::_SENTINEL - 1})[0];
			size_t nbytes = shape.n_elems() * clay::type_size(dtype);
			std::shared_ptr<char> data = clay::make_char(nbytes);
			op->set_args({
				clay::State(data, shape, dtype),
				clay::State(data, shape, dtype)});
		}
		break;
		case slip::BINO:
		{
			clay::Shape shape = random_def_shape(fuzzer);
			size_t nbytes0 = shape.n_elems() * sizeof(uint64_t);
			size_t nbytes1 = shape.n_elems() * sizeof(double);
			std::shared_ptr<char> data0 = clay::make_char(nbytes0);
			std::shared_ptr<char> data1 = clay::make_char(nbytes1);
			op->set_args({
				clay::State(data0, shape, clay::UINT64),
				clay::State(data1, shape, clay::DOUBLE)});
		}
		break;
		case slip::NORM:
		{
			clay::Shape shape = random_def_shape(fuzzer);
			clay::DTYPE dtype = clay::DOUBLE;
			size_t nbytes = shape.n_elems() * clay::type_size(dtype);
			std::shared_ptr<char> data = clay::make_char(nbytes);
			op->set_args({
				clay::State(data, shape, dtype),
				clay::State(data, shape, dtype)});
		}
		break;
		case slip::FLIP:
		case slip::N_DIMS:
		case slip::ARGMAX:
		case slip::RMAX:
		case slip::RSUM:
		{
			clay::Shape shape = random_def_shape(fuzzer);
			clay::Shape wun({1});
			clay::DTYPE dtype = (clay::DTYPE)
				fuzzer->get_int(1, "dtype", {1, clay::_SENTINEL - 1})[0];
			size_t nbytes = shape.n_elems() * clay::type_size(dtype);
			std::shared_ptr<char> data = clay::make_char(nbytes);
			std::shared_ptr<char> s = clay::make_char(sizeof(uint64_t));
			std::memset(s.get(), 0, sizeof(uint64_t));
			op->set_args({
				clay::State(data, shape, dtype),
				clay::State(s, wun, clay::UINT64)});
		}
		break;
		case slip::EXPAND:
		{
			clay::Shape shape = random_def_shape(fuzzer);
			clay::Shape wun({1});
			clay::DTYPE dtype = (clay::DTYPE)
				fuzzer->get_int(1, "dtype", {1, clay::_SENTINEL - 1})[0];
			size_t nbytes = shape.n_elems() * clay::type_size(dtype);
			std::shared_ptr<char> data = clay::make_char(nbytes);
			std::shared_ptr<char> s = clay::make_char(sizeof(uint64_t));
			std::memset(s.get(), 0, sizeof(uint64_t));
			op->set_args({
				clay::State(data, shape, dtype),
				clay::State(s, wun, clay::UINT64),
				clay::State(s, wun, clay::UINT64)});
		}
		break;
		case slip::MATMUL:
		{
			std::vector<size_t> clist = random_def_shape(fuzzer); // <m, n, ...>
			size_t k = fuzzer->get_int(1, "k", {1, 8})[0];
			size_t m = clist[0];
			clay::Shape shape0 = clist;
			clist[1] = m;
			clist[0] = k;
			clay::Shape shape1 = clist; // <k, m, ...>
			clay::DTYPE dtype = (clay::DTYPE)
				fuzzer->get_int(1, "dtype", {1, clay::_SENTINEL - 1})[0];
			size_t nbytes0 = shape0.n_elems() * clay::type_size(dtype);
			size_t nbytes1 = shape1.n_elems() * clay::type_size(dtype);
			std::shared_ptr<char> data0 = clay::make_char(nbytes0);
			std::shared_ptr<char> data1 = clay::make_char(nbytes1);
			op->set_args({
				clay::State(data0, shape0, dtype),
				clay::State(data1, shape1, dtype)});
		}
		break;
		default:
		break;
	}
}


TEST_F(OPERATE_IO, SetEmpty_A000)
{
	slip::OPCODE opcode = (slip::OPCODE) get_int(1, "opcode", {0, slip::MATMUL})[0];
	mold::iOperatePtrT op = slip::get_op(opcode);
	EXPECT_THROW(op->set_args({}), slip::NoArgumentsError);
}


TEST_F(OPERATE_IO, SetOnce_A001)
{
	slip::OPCODE opcode = (slip::OPCODE) get_int(1, "opcode", {0, slip::MATMUL})[0];
	mold::iOperatePtrT op = slip::get_op(opcode);
	set_op(this, op, opcode);
	EXPECT_THROW(set_op(this, op, opcode), std::bad_function_call);
}


TEST_F(OPERATE_IO, ReadBeforeSet_A002)
{
	slip::OPCODE opcode = (slip::OPCODE) get_int(1, "opcode", {0, slip::MATMUL})[0];
	mold::iOperatePtrT op = slip::get_op(opcode);
	clay::State out;
	EXPECT_THROW(op->read_data(out), std::bad_function_call);
}


TEST_F(OPERATE_IO, GetImmsBeforeSet_A003)
{
	slip::OPCODE opcode = (slip::OPCODE) get_int(1, "opcode", {0, slip::MATMUL})[0];
	mold::iOperatePtrT op = slip::get_op(opcode);
	EXPECT_THROW(op->get_imms(), slip::NoArgumentsError);
}


#endif /* DISABLE_OPERATE_IO_TEST */


#endif /* DISABLE_SLIP_MODULE_TESTS */
