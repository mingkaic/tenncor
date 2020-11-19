#include <string>
#include <cstring>
#include "internal/global/global.hpp"


#ifndef _GENERATED_DTYPES_HPP
#define _GENERATED_DTYPES_HPP

namespace egen
{

#define EGEN_FULLTYPE

enum _GENERATED_DTYPE
{
    BAD_TYPE = 0,
    //>>> enumeration
    DOUBLE,
    FLOAT,
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    UINT32,
    INT64,
    UINT64,
    _N_GENERATED_DTYPES,
};

const _GENERATED_DTYPE default_dtype = FLOAT;

std::string name_type (_GENERATED_DTYPE type);

uint8_t type_size (_GENERATED_DTYPE type);

size_t type_precision (_GENERATED_DTYPE type);

_GENERATED_DTYPE get_type (const std::string& name);

template <typename T>
_GENERATED_DTYPE get_type (void)
{
    return BAD_TYPE;
}

template <typename T>
struct TypeInfo
{
    static const _GENERATED_DTYPE type = BAD_TYPE;

    TypeInfo (void) = delete;
};

//>>> mapping
template <>
_GENERATED_DTYPE get_type<double> (void);

template <>
struct TypeInfo<double>
{
    static const _GENERATED_DTYPE type = DOUBLE;

    TypeInfo (void) = delete;
};

template <>
_GENERATED_DTYPE get_type<float> (void);

template <>
struct TypeInfo<float>
{
    static const _GENERATED_DTYPE type = FLOAT;

    TypeInfo (void) = delete;
};

template <>
_GENERATED_DTYPE get_type<int8_t> (void);

template <>
struct TypeInfo<int8_t>
{
    static const _GENERATED_DTYPE type = INT8;

    TypeInfo (void) = delete;
};

template <>
_GENERATED_DTYPE get_type<uint8_t> (void);

template <>
struct TypeInfo<uint8_t>
{
    static const _GENERATED_DTYPE type = UINT8;

    TypeInfo (void) = delete;
};

template <>
_GENERATED_DTYPE get_type<int16_t> (void);

template <>
struct TypeInfo<int16_t>
{
    static const _GENERATED_DTYPE type = INT16;

    TypeInfo (void) = delete;
};

template <>
_GENERATED_DTYPE get_type<uint16_t> (void);

template <>
struct TypeInfo<uint16_t>
{
    static const _GENERATED_DTYPE type = UINT16;

    TypeInfo (void) = delete;
};

template <>
_GENERATED_DTYPE get_type<int32_t> (void);

template <>
struct TypeInfo<int32_t>
{
    static const _GENERATED_DTYPE type = INT32;

    TypeInfo (void) = delete;
};

template <>
_GENERATED_DTYPE get_type<uint32_t> (void);

template <>
struct TypeInfo<uint32_t>
{
    static const _GENERATED_DTYPE type = UINT32;

    TypeInfo (void) = delete;
};

template <>
_GENERATED_DTYPE get_type<int64_t> (void);

template <>
struct TypeInfo<int64_t>
{
    static const _GENERATED_DTYPE type = INT64;

    TypeInfo (void) = delete;
};

template <>
_GENERATED_DTYPE get_type<uint64_t> (void);

template <>
struct TypeInfo<uint64_t>
{
    static const _GENERATED_DTYPE type = UINT64;

    TypeInfo (void) = delete;
};

// converts from input to output type
template <typename OUTTYPE>
void type_convert (OUTTYPE* out, const void* input,
    _GENERATED_DTYPE intype, size_t nelems)
{
    switch (intype)
    {
        //>>> conversions
        case DOUBLE:
        {
            std::vector<OUTTYPE> temp((double*) input, (double*) input + nelems);
            std::memcpy(out, temp.data(), sizeof(OUTTYPE) * nelems);
        }
            break;
        case FLOAT:
        {
            std::vector<OUTTYPE> temp((float*) input, (float*) input + nelems);
            std::memcpy(out, temp.data(), sizeof(OUTTYPE) * nelems);
        }
            break;
        case INT8:
        {
            std::vector<OUTTYPE> temp((int8_t*) input, (int8_t*) input + nelems);
            std::memcpy(out, temp.data(), sizeof(OUTTYPE) * nelems);
        }
            break;
        case UINT8:
        {
            std::vector<OUTTYPE> temp((uint8_t*) input, (uint8_t*) input + nelems);
            std::memcpy(out, temp.data(), sizeof(OUTTYPE) * nelems);
        }
            break;
        case INT16:
        {
            std::vector<OUTTYPE> temp((int16_t*) input, (int16_t*) input + nelems);
            std::memcpy(out, temp.data(), sizeof(OUTTYPE) * nelems);
        }
            break;
        case UINT16:
        {
            std::vector<OUTTYPE> temp((uint16_t*) input, (uint16_t*) input + nelems);
            std::memcpy(out, temp.data(), sizeof(OUTTYPE) * nelems);
        }
            break;
        case INT32:
        {
            std::vector<OUTTYPE> temp((int32_t*) input, (int32_t*) input + nelems);
            std::memcpy(out, temp.data(), sizeof(OUTTYPE) * nelems);
        }
            break;
        case UINT32:
        {
            std::vector<OUTTYPE> temp((uint32_t*) input, (uint32_t*) input + nelems);
            std::memcpy(out, temp.data(), sizeof(OUTTYPE) * nelems);
        }
            break;
        case INT64:
        {
            std::vector<OUTTYPE> temp((int64_t*) input, (int64_t*) input + nelems);
            std::memcpy(out, temp.data(), sizeof(OUTTYPE) * nelems);
        }
            break;
        case UINT64:
        {
            std::vector<OUTTYPE> temp((uint64_t*) input, (uint64_t*) input + nelems);
            std::memcpy(out, temp.data(), sizeof(OUTTYPE) * nelems);
        }
            break;
        default:
            global::fatalf("invalid input type %s",
                name_type(intype).c_str());
    }
}

#define EVERY_TYPE(GENERIC_MACRO)\
GENERIC_MACRO(egen::DOUBLE,double)\
GENERIC_MACRO(egen::FLOAT,float)\
GENERIC_MACRO(egen::INT8,int8_t)\
GENERIC_MACRO(egen::UINT8,uint8_t)\
GENERIC_MACRO(egen::INT16,int16_t)\
GENERIC_MACRO(egen::UINT16,uint16_t)\
GENERIC_MACRO(egen::INT32,int32_t)\
GENERIC_MACRO(egen::UINT32,uint32_t)\
GENERIC_MACRO(egen::INT64,int64_t)\
GENERIC_MACRO(egen::UINT64,uint64_t)

// GENERIC_MACRO must accept a real type as an argument.
// e.g.:
// #define GENERIC_MACRO(REAL_TYPE) run<REAL_TYPE>(args...);
// ...
// TYPE_LOOKUP(GENERIC_MACRO, type_code)
#define TYPE_LOOKUP(GENERIC_MACRO, DTYPE)\
switch (DTYPE)\
{\
    case egen::DOUBLE: GENERIC_MACRO(double) break;\
    case egen::FLOAT: GENERIC_MACRO(float) break;\
    case egen::INT8: GENERIC_MACRO(int8_t) break;\
    case egen::UINT8: GENERIC_MACRO(uint8_t) break;\
    case egen::INT16: GENERIC_MACRO(int16_t) break;\
    case egen::UINT16: GENERIC_MACRO(uint16_t) break;\
    case egen::INT32: GENERIC_MACRO(int32_t) break;\
    case egen::UINT32: GENERIC_MACRO(uint32_t) break;\
    case egen::INT64: GENERIC_MACRO(int64_t) break;\
    case egen::UINT64: GENERIC_MACRO(uint64_t) break;\
    default: global::fatal("executing bad type");\
}
//>>> ^ cases

}

#endif // _GENERATED_DTYPES_HPP
