#include "sand/type.hpp"
#include "util/error.hpp"
#include "util/mapper.hpp"

#ifdef SAND_TYPE_HPP

#define TYPE_ASSOC(TYPE) std::pair<DTYPE,std::string>{TYPE, #TYPE}

using TypenameMap = EnumMap<DTYPE,std::string>;

const TypenameMap named_types =
{
	TYPE_ASSOC(BAD),
	TYPE_ASSOC(DOUBLE),
	TYPE_ASSOC(FLOAT),
	TYPE_ASSOC(INT8),
	TYPE_ASSOC(INT16),
	TYPE_ASSOC(INT32),
	TYPE_ASSOC(INT64),
	TYPE_ASSOC(UINT8),
	TYPE_ASSOC(UINT16),
	TYPE_ASSOC(UINT32),
	TYPE_ASSOC(UINT64),
	std::pair<DTYPE,std::string>{_SENTINEL, ""}
};

std::string name_type (DTYPE type)
{
	auto it = named_types.find(type);
	if (named_types.end() == it)
	{
		return "BAD_TYPE";
	}
	return it->second;
}

template <>
DTYPE get_type<double> (void)
{
	return DTYPE::DOUBLE;
}

template <>
DTYPE get_type<float> (void)
{
	return DTYPE::FLOAT;
}

template <>
DTYPE get_type<int8_t> (void)
{
	return DTYPE::INT8;
}

template <>
DTYPE get_type<uint8_t> (void)
{
	return DTYPE::UINT8;
}

template <>
DTYPE get_type<int16_t> (void)
{
	return DTYPE::INT16;
}

template <>
DTYPE get_type<uint16_t> (void)
{
	return DTYPE::UINT16;
}

template <>
DTYPE get_type<int32_t> (void)
{
	return DTYPE::INT32;
}

template <>
DTYPE get_type<uint32_t> (void)
{
	return DTYPE::UINT32;
}

template <>
DTYPE get_type<int64_t> (void)
{
	return DTYPE::INT64;
}

template <>
DTYPE get_type<uint64_t> (void)
{
	return DTYPE::UINT64;
}

uint8_t type_size (DTYPE type)
{
	switch (type)
	{
		case DTYPE::DOUBLE:
			return sizeof(double);
		case DTYPE::FLOAT:
			return sizeof(float);
		case DTYPE::INT8:
		case DTYPE::UINT8:
			return sizeof(int8_t);
		case DTYPE::INT16:
		case DTYPE::UINT16:
			return sizeof(int16_t);
		case DTYPE::INT32:
		case DTYPE::UINT32:
			return sizeof(int32_t);
		case DTYPE::INT64:
		case DTYPE::UINT64:
			return sizeof(int64_t);
		default:
			handle_error("unsupported type",
				ErrArg<size_t>("typeval", type));
	}
	return 0;
}

#endif
