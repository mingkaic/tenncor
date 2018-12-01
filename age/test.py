import unittest
import difflib

import age.templates.api_tmpl as api
import age.templates.capi_tmpl as capi
import age.templates.codes_tmpl as codes
import age.templates.grader_tmpl as grader
import age.templates.opera_tmpl as opera

api_fields = {"apis": [
    {"name": "func1", "args": [], "out": "bar1()"},
    {"name": "func2", "args": ["ade::TensptrT arg", "Arg arg1"], "out": "bar2()"},
    {"name": "func3", "args": ["ade::TensptrT arg", "Arg arg1", "ade::TensptrT arg2"], "out": "bar3()"}
]}

codes_fields = {
    "opcodes": ["OP", "OP1", "OP2", "OP3"],
    "dtypes": {
        "CAR": "char",
        "VROOM": "double",
        "VRUM": "float",
        "KAPOW": "complex_t"
    }
}

grader_fields = {
    "scalarize": "get_numba(12345)",
    "sum": "ADDITION",
    "prod": "MULTIPLICATION",
    "grads": {
        "FOO": "bwd_foo(args, idx)",
        "BAR": "bwd_bar(args[0], idx)",
        "BAR2": "foo(args[1])",
    }
}

opera_fields = {
    "data_in": "In_Type",
    "data_out": "Out_Type",
    "ops": {
        "OP": "foo(out, shape)",
        "OP2": "bar1(out, shape, in)",
        "FOO": "bar2(out, in[1])",
    },
    "types": {
        "APPLE": "skin_t",
        "BANANA": "peel_t",
        "ORANGE": "seed_t",
        "LEMON": "slices_t"
    }
}

api_header = """#ifndef _GENERATED_API_HPP
#define _GENERATED_API_HPP

namespace age
{

ade::TensptrT func1 ();

ade::TensptrT func2 (ade::TensptrT arg, Arg arg1);

ade::TensptrT func3 (ade::TensptrT arg, Arg arg1, ade::TensptrT arg2);

}

#endif // _GENERATED_API_HPP
"""

api_source = """#ifdef _GENERATED_API_HPP

namespace age
{

ade::TensptrT func1 ()
{
    if (false)
    {
        logs::fatal("cannot func1 with a null argument");
    }
    return bar1();
}

ade::TensptrT func2 (ade::TensptrT arg, Arg arg1)
{
    if (arg == nullptr)
    {
        logs::fatal("cannot func2 with a null argument");
    }
    return bar2();
}

ade::TensptrT func3 (ade::TensptrT arg, Arg arg1, ade::TensptrT arg2)
{
    if (arg == nullptr || arg2 == nullptr)
    {
        logs::fatal("cannot func3 with a null argument");
    }
    return bar3();
}

}

#endif
"""

capi_header = """#ifndef _GENERATED_CAPI_HPP
#define _GENERATED_CAPI_HPP

int64_t register_tens (ade::iTensor* ptr);

int64_t register_tens (ade::TensptrT& ptr);

ade::TensptrT get_tens (int64_t id);

extern void free_tens (int64_t id);

extern void get_shape (int outshape[8], int64_t tens);

extern int64_t age_func1 ();

extern int64_t age_func2 (int64_t arg, Arg arg1);

extern int64_t age_func3 (int64_t arg, Arg arg1, int64_t arg2);

#endif // _GENERATED_CAPI_HPP
"""

capi_source = """#ifdef _GENERATED_CAPI_HPP

static std::unordered_map<int64_t,ade::TensptrT> tens;

int64_t register_tens (ade::iTensor* ptr)
{
    int64_t id = (int64_t) ptr;
    tens.emplace(id, ade::TensptrT(ptr));
    return id;
}

int64_t register_tens (ade::TensptrT& ptr)
{
    int64_t id = (int64_t) ptr.get();
    tens.emplace(id, ptr);
    return id;
}

ade::TensptrT get_tens (int64_t id)
{
    auto it = tens.find(id);
    if (tens.end() == it)
    {
        return ade::TensptrT(nullptr);
    }
    return it->second;
}

void free_tens (int64_t id)
{
    tens.erase(id);
}

void get_shape (int outshape[8], int64_t id)
{
    const ade::Shape& shape = get_tens(id)->shape();
    std::copy(shape.begin(), shape.end(), outshape);
}

int64_t age_func1 ()
{
    auto ptr = age::func1();
    int64_t id = (int64_t) ptr.get();
    tens.emplace(id, ptr);
    return id;
}

int64_t age_func2 (int64_t arg, Arg arg1)
{
    ade::TensptrT arg_ptr = get_tens(arg);
    auto ptr = age::func2(arg_ptr, arg1);
    int64_t id = (int64_t) ptr.get();
    tens.emplace(id, ptr);
    return id;
}

int64_t age_func3 (int64_t arg, Arg arg1, int64_t arg2)
{
    ade::TensptrT arg_ptr = get_tens(arg);
    ade::TensptrT arg2_ptr = get_tens(arg2);
    auto ptr = age::func3(arg_ptr, arg1, arg2_ptr);
    int64_t id = (int64_t) ptr.get();
    tens.emplace(id, ptr);
    return id;
}

#endif
"""

codes_header = """#ifndef _GENERATED_CODES_HPP
#define _GENERATED_CODES_HPP

namespace age
{

enum _GENERATED_OPCODE
{
    BAD_OP = 0,
    OP,
    OP1,
    OP2,
    OP3,
};

enum _GENERATED_DTYPE
{
    BAD_TYPE = 0,
    CAR,
    VRUM,
    VROOM,
    KAPOW,
};

std::string name_op (_GENERATED_OPCODE code);

_GENERATED_OPCODE get_op (std::string name);

std::string name_type (_GENERATED_DTYPE type);

uint8_t type_size (_GENERATED_DTYPE type);

_GENERATED_DTYPE get_type (std::string name);

template <typename T>
_GENERATED_DTYPE get_type (void)
{
    return BAD_TYPE;
}

template <>
_GENERATED_DTYPE get_type<char> (void);

template <>
_GENERATED_DTYPE get_type<float> (void);

template <>
_GENERATED_DTYPE get_type<double> (void);

template <>
_GENERATED_DTYPE get_type<complex_t> (void);

}

#endif // _GENERATED_CODES_HPP
"""

codes_source = """#ifdef _GENERATED_CODES_HPP

namespace age
{

struct EnumHash
{
    template <typename T>
    size_t operator() (T e) const
    {
        return static_cast<size_t>(e);
    }
};

static std::unordered_map<_GENERATED_OPCODE,std::string,EnumHash> code2name =
{
    { OP, "OP" },
    { OP1, "OP1" },
    { OP2, "OP2" },
    { OP3, "OP3" },
};

static std::unordered_map<std::string,_GENERATED_OPCODE> name2code =
{
    { "OP", OP },
    { "OP1", OP1 },
    { "OP2", OP2 },
    { "OP3", OP3 },
};

static std::unordered_map<_GENERATED_DTYPE,std::string,EnumHash> type2name =
{
    { CAR, "CAR" },
    { VRUM, "VRUM" },
    { VROOM, "VROOM" },
    { KAPOW, "KAPOW" },
};

static std::unordered_map<std::string,_GENERATED_DTYPE> name2type =
{
    { "CAR", CAR },
    { "VRUM", VRUM },
    { "VROOM", VROOM },
    { "KAPOW", KAPOW },
};

std::string name_op (_GENERATED_OPCODE code)
{
    auto it = code2name.find(code);
    if (code2name.end() == it)
    {
        return "BAD_OP";
    }
    return it->second;
}

_GENERATED_OPCODE get_op (std::string name)
{
    auto it = name2code.find(name);
    if (name2code.end() == it)
    {
        return BAD_OP;
    }
    return it->second;
}

std::string name_type (_GENERATED_DTYPE type)
{
    auto it = type2name.find(type);
    if (type2name.end() == it)
    {
        return "BAD_TYPE";
    }
    return it->second;
}

_GENERATED_DTYPE get_type (std::string name)
{
    auto it = name2type.find(name);
    if (name2type.end() == it)
    {
        return BAD_TYPE;
    }
    return it->second;
}

uint8_t type_size (_GENERATED_DTYPE type)
{
    switch (type)
    {
        case CAR: return sizeof(char);
        case VRUM: return sizeof(float);
        case VROOM: return sizeof(double);
        case KAPOW: return sizeof(complex_t);
        default: logs::fatal("cannot get size of bad type");
    }
}

template <>
_GENERATED_DTYPE get_type<char> (void)
{
    return CAR;
}

template <>
_GENERATED_DTYPE get_type<float> (void)
{
    return VRUM;
}

template <>
_GENERATED_DTYPE get_type<double> (void)
{
    return VROOM;
}

template <>
_GENERATED_DTYPE get_type<complex_t> (void)
{
    return KAPOW;
}

}

#endif
"""

grader_header = """#ifndef _GENERATED_GRADER_HPP
#define _GENERATED_GRADER_HPP

namespace age
{

template <typename T>
ade::LeafptrT data (T scalar, ade::Shape shape)
{
    return get_numba(12345);
}

struct RuleSet final : public iRuleSet
{
    ade::LeafptrT data (double scalar, ade::Shape shape) override
    {
        return age::data(scalar, shape);
    }

    ade::Opcode sum_opcode (void) override
    {
        return ade::Opcode{"ADDITION", ADDITION};
    }

    ade::Opcode prod_opcode (void) override
    {
        return ade::Opcode{"MULTIPLICATION", MULTIPLICATION};
    }

    ade::TensptrT grad_rule (size_t code, TensT args, size_t idx) override;
};

}

#endif // _GENERATED_GRADER_HPP
"""

grader_source = """#ifdef _GENERATED_GRADER_HPP

namespace age
{

ade::TensptrT RuleSet::grad_rule (size_t code,TensT args,size_t idx)
{
    switch (code)
    {
        case FOO: return bwd_foo(args, idx);
        case BAR: return bwd_bar(args[0], idx);
        case BAR2: return foo(args[1]);
        default: logs::fatal("no gradient rule for unknown opcode");
    }
}

}

#endif
"""

opera_header = """#ifndef _GENERATED_OPERA_HPP
#define _GENERATED_OPERA_HPP

namespace age
{

template <typename T>
void typed_exec (_GENERATED_OPCODE opcode,
    Out_Type out, ade::Shape shape, In_Type in)
{
    switch (opcode)
    {
        case FOO:
            bar2(out, in[1]); break;
        case OP2:
            bar1(out, shape, in); break;
        case OP:
            foo(out, shape); break;
        default: logs::fatal("unknown opcode");
    }
}

void op_exec (_GENERATED_OPCODE opcode, _GENERATED_DTYPE dtype,
    Out_Type out, ade::Shape shape, In_Type in);

}

#endif // _GENERATED_OPERA_HPP
"""

opera_source = """#ifdef _GENERATED_OPERA_HPP

namespace age
{

void op_exec (_GENERATED_OPCODE opcode, _GENERATED_DTYPE dtype,
    Out_Type out, ade::Shape shape, In_Type in)
{
    switch (dtype)
    {
        case ORANGE:
            typed_exec<seed_t>(opcode, out, shape, in); break;
        case LEMON:
            typed_exec<slices_t>(opcode, out, shape, in); break;
        case APPLE:
            typed_exec<skin_t>(opcode, out, shape, in); break;
        case BANANA:
            typed_exec<peel_t>(opcode, out, shape, in); break;
        default: logs::fatal("executing bad type");
    }
}

}

#endif
"""

def multiline_check(s1, s2):
    arr1 = list(filter(lambda line: len(line) > 0, s1.splitlines()))
    arr2 = list(filter(lambda line: len(line) > 0, s2.splitlines()))

    diffstr = '\n'.join(difflib.unified_diff(arr1, arr2))
    diffs = diffstr.splitlines()
    if len(diffs) > 0:
        print(diffstr)

class ClientTest(unittest.TestCase):
    def test_api(self):
        header = api.header.repr(api_fields)
        source = api.source.repr(api_fields)
        multiline_check(api_header, header)
        multiline_check(api_source, source)
        self.assertEqual(api_header, header)
        self.assertEqual(api_source, source)

    def test_capi(self):
        header = capi.header.repr(api_fields)
        source = capi.source.repr(api_fields)
        multiline_check(capi_header, header)
        multiline_check(capi_source, source)
        self.assertEqual(capi_header, header)
        self.assertEqual(capi_source, source)

    def test_codes(self):
        header = codes.header.repr(codes_fields)
        source = codes.source.repr(codes_fields)
        multiline_check(codes_header, header)
        multiline_check(codes_source, source)
        self.assertEqual(codes_header, header)
        self.assertEqual(codes_source, source)

    def test_grader(self):
        header = grader.header.repr(grader_fields)
        source = grader.source.repr(grader_fields)
        multiline_check(grader_header, header)
        multiline_check(grader_source, source)
        self.assertEqual(grader_header, header)
        self.assertEqual(grader_source, source)

    def test_opera(self):
        header = opera.header.repr(opera_fields)
        source = opera.source.repr(opera_fields)
        multiline_check(opera_header, header)
        multiline_check(opera_source, source)
        self.assertEqual(opera_header, header)
        self.assertEqual(opera_source, source)

if __name__ == "__main__":
    unittest.main()
