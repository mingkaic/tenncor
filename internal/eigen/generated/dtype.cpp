#include "estd/contain.hpp"
#include "internal/eigen/generated/dtype.hpp"


#ifdef _GENERATED_DTYPES_HPP

namespace egen
{

static const std::unordered_map<_GENERATED_DTYPE,std::string,estd::EnumHash> type2name =
{
    //>>> type2names
    { DOUBLE, "DOUBLE" },
    { FLOAT, "FLOAT" },
    { INT8, "INT8" },
    { UINT8, "UINT8" },
    { INT16, "INT16" },
    { UINT16, "UINT16" },
    { INT32, "INT32" },
    { UINT32, "UINT32" },
    { INT64, "INT64" },
    { UINT64, "UINT64" }
};

static const std::unordered_map<std::string,_GENERATED_DTYPE> name2type =
{
    //>>> name2types
    { "DOUBLE", DOUBLE },
    { "FLOAT", FLOAT },
    { "INT8", INT8 },
    { "UINT8", UINT8 },
    { "INT16", INT16 },
    { "UINT16", UINT16 },
    { "INT32", INT32 },
    { "UINT32", UINT32 },
    { "INT64", INT64 },
    { "UINT64", UINT64 }
};

std::string name_type (_GENERATED_DTYPE type)
{
    return estd::try_get(type2name, type, "BAD_DTYPE");
}

_GENERATED_DTYPE get_type (const std::string& name)
{
    return estd::try_get(name2type, name, BAD_TYPE);
}

uint8_t type_size (_GENERATED_DTYPE type)
{
    switch (type)
    {
        //>>> typesizes
        case egen::DOUBLE: return sizeof(double);
    case egen::FLOAT: return sizeof(float);
    case egen::INT8: return sizeof(int8_t);
    case egen::UINT8: return sizeof(uint8_t);
    case egen::INT16: return sizeof(int16_t);
    case egen::UINT16: return sizeof(uint16_t);
    case egen::INT32: return sizeof(int32_t);
    case egen::UINT32: return sizeof(uint32_t);
    case egen::INT64: return sizeof(int64_t);
    case egen::UINT64: return sizeof(uint64_t);
        default: global::fatal("cannot get size of bad type");
    }
    return 0;
}


size_t type_precision (_GENERATED_DTYPE type)
{
    switch (type)
    {
        //>>> precisions
        case egen::DOUBLE: return 10;
    case egen::FLOAT: return 9;
    case egen::INT8: return 1;
    case egen::UINT8: return 2;
    case egen::INT16: return 3;
    case egen::UINT16: return 4;
    case egen::INT32: return 5;
    case egen::UINT32: return 6;
    case egen::INT64: return 7;
    case egen::UINT64: return 8;
        default: break;
    }
    return 0;
}

//>>> get_types

template <>
_GENERATED_DTYPE get_type<double> (void)
{
    return DOUBLE;
}

template <>
_GENERATED_DTYPE get_type<float> (void)
{
    return FLOAT;
}

template <>
_GENERATED_DTYPE get_type<int8_t> (void)
{
    return INT8;
}

template <>
_GENERATED_DTYPE get_type<uint8_t> (void)
{
    return UINT8;
}

template <>
_GENERATED_DTYPE get_type<int16_t> (void)
{
    return INT16;
}

template <>
_GENERATED_DTYPE get_type<uint16_t> (void)
{
    return UINT16;
}

template <>
_GENERATED_DTYPE get_type<int32_t> (void)
{
    return INT32;
}

template <>
_GENERATED_DTYPE get_type<uint32_t> (void)
{
    return UINT32;
}

template <>
_GENERATED_DTYPE get_type<int64_t> (void)
{
    return INT64;
}

template <>
_GENERATED_DTYPE get_type<uint64_t> (void)
{
    return UINT64;
}


}

#endif
