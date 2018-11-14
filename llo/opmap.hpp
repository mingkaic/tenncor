///
/// opmap.hpp
/// llo
///
/// Purpose:
/// Associate age::OPCODE to operations on data
///

#include "llo/generated/runtime.hpp"

#include "llo/data.hpp"
#include "llo/helper.hpp"

#ifndef LLO_OPMAP_HPP
#define LLO_OPMAP_HPP

namespace llo
{

template <typename T>
void typed_exec (age::_GENERATED_OPCODES opcode, GenericData& out, DataArgsT& data)
{
	switch (opcode)
	{
		case age::ABS: abs((T*) out.data_.get(), to_ref<T>(data[0])); break;
		case age::NEG: neg((T*) out.data_.get(), to_ref<T>(data[0])); break;
		case age::SIN: sin((T*) out.data_.get(), to_ref<T>(data[0])); break;
		case age::COS: cos((T*) out.data_.get(), to_ref<T>(data[0])); break;
		case age::TAN: tan((T*) out.data_.get(), to_ref<T>(data[0])); break;
		case age::EXP: exp((T*) out.data_.get(), to_ref<T>(data[0])); break;
		case age::LOG: log((T*) out.data_.get(), to_ref<T>(data[0])); break;
		case age::SQRT: sqrt((T*) out.data_.get(), to_ref<T>(data[0])); break;
		case age::ROUND: round((T*) out.data_.get(), to_ref<T>(data[0])); break;
		case age::POW: pow((T*) out.data_.get(), out.shape_, to_ref<T>(data[0]), to_ref<T>(data[1])); break;
		case age::SUM: add((T*) out.data_.get(), out.shape_, to_refs<T>(data)); break;
		case age::SUB: sub((T*) out.data_.get(), out.shape_, to_ref<T>(data[0]), to_ref<T>(data[1])); break;
		case age::PROD: mul((T*) out.data_.get(), out.shape_, to_refs<T>(data)); break;
		case age::DIV: div((T*) out.data_.get(), out.shape_, to_ref<T>(data[0]), to_ref<T>(data[1])); break;
		case age::EQ: eq((T*) out.data_.get(), out.shape_, to_ref<T>(data[0]), to_ref<T>(data[1])); break;
		case age::NEQ: neq((T*) out.data_.get(), out.shape_, to_ref<T>(data[0]), to_ref<T>(data[1])); break;
		case age::LT: lt((T*) out.data_.get(), out.shape_, to_ref<T>(data[0]), to_ref<T>(data[1])); break;
		case age::GT: gt((T*) out.data_.get(), out.shape_, to_ref<T>(data[0]), to_ref<T>(data[1])); break;
		case age::MIN: min((T*) out.data_.get(), out.shape_, to_refs<T>(data)); break;
		case age::MAX: max((T*) out.data_.get(), out.shape_, to_refs<T>(data)); break;
		case age::RAND_BINO: rand_binom((T*) out.data_.get(), out.shape_, to_ref<T>(data[0]), to_ref<double>(data[1])); break;
		case age::RAND_UNIF: rand_uniform((T*) out.data_.get(), out.shape_, to_ref<T>(data[0]), to_ref<T>(data[1])); break;
		case age::RAND_NORM: rand_normal((T*) out.data_.get(), out.shape_, to_ref<T>(data[0]), to_ref<T>(data[1])); break;
		default: err::fatal("unknown opcode");
	}
}

void op_exec (age::_GENERATED_OPCODES opcode, GenericData& out, DataArgsT& data);

}

#endif // LLO_OPMAP_HPP
