///
/// serialize.hpp
/// eteq
///
/// Purpose:
/// Define functions for marshal and unmarshal data sources
///

#include "pbm/data.hpp"

#include "eteq/generated/opcode.hpp"
#include "eteq/generated/dtype.hpp"

#include "eteq/coord.hpp"
#include "eteq/constant.hpp"
#include "eteq/variable.hpp"
#include "eteq/functor.hpp"

#ifndef ETEQ_SERIALIZE_HPP
#define ETEQ_SERIALIZE_HPP

namespace eteq
{

static std::unordered_set<size_t> non_bijectives =
{
	egen::REDUCE_SUM,
	egen::REDUCE_PROD,
	egen::REDUCE_MIN,
	egen::REDUCE_MAX,
	egen::EXTEND,
};

static bool is_big_endian(void)
{
	union
	{
		uint16_t _;
		char bytes[2];
	} twob = { 0x0001 };

	return twob.bytes[0] == 0;
}

/// PBM Marshaller implementation for saving ETEQ Nodes
struct EADSaver final : public pbm::iSaver
{
	/// Implementation of iSaver
	std::string save_leaf (teq::iLeaf* leaf) override
	{
		char* data = (char*) leaf->data();
		size_t nelems = leaf->shape().n_elems();
		size_t nbytes = egen::type_size((egen::_GENERATED_DTYPE) leaf->type_code());
		if (is_big_endian() && nbytes > 1)
		{
			size_t totalbytes = nelems * nbytes;
			std::string out(totalbytes, '\0');
			for (size_t i = 0; i < totalbytes; ++i)
			{
				size_t elemi = i / nbytes;
				size_t outi = (elemi + 1) * nbytes - (i % nbytes);
				out[outi] = data[i];
			}
			return out;
		}
		return std::string(data, nelems * nbytes);
	}

	/// Implementation of iSaver
	std::vector<double> save_shaper (const teq::CoordptrT& mapper) override
	{
		std::vector<double> out;
		mapper->access(
			[&out](const teq::MatrixT& mat)
			{
				for (teq::RankT i = 0; i < teq::mat_dim; ++i)
				{
					for (teq::RankT j = 0; j < teq::mat_dim; ++j)
					{
						out.push_back(mat[i][j]);
					}
				}
			});
		return out;
	}

	/// Implementation of iSaver
	std::vector<double> save_coorder (const teq::CoordptrT& mapper) override
	{
		if (nullptr == mapper)
		{
			return std::vector<double>();
		}
		teq::CoordT coord;
		mapper->forward(coord.begin(), coord.begin());
		return std::vector<double>(coord.begin(), coord.end());
	}
};

#define _OUT_GENERIC(realtype)leaf = is_const?\
teq::TensptrT(Constant<realtype>::get((realtype*) pb, shape)):\
teq::TensptrT(Variable<realtype>::get((realtype*) pb, shape, label));

#define _OUT_GENFUNC(realtype){\
ArgsT<realtype> eargs;eargs.reserve(args.size());\
std::transform(args.begin(), args.end(), std::back_inserter(eargs),\
[](teq::FuncArg arg){\
	return FuncArg<realtype>(\
		NodeConverters<realtype>::to_node(arg.get_tensor()),\
		arg.get_shaper(),\
		std::static_pointer_cast<CoordMap>(arg.get_coorder()));\
});\
func = teq::TensptrT(\
Functor<realtype>::get(teq::Opcode{opname, egen::get_op(opname)},eargs));}

/// PBM Unmarshaller implementation for loading ETEQ Nodes
struct EADLoader final : public pbm::iLoader
{
	/// Implementation of iLoader
	teq::TensptrT generate_leaf (const char* pb, teq::Shape shape,
		std::string typelabel, std::string label, bool is_const) override
	{
		teq::TensptrT leaf;
		egen::_GENERATED_DTYPE gencode = egen::get_type(typelabel);
		size_t nbytes = egen::type_size(gencode);
		if (is_big_endian() && nbytes > 1)
		{
			size_t totalbytes = shape.n_elems() * nbytes;
			std::string out(totalbytes, '\0');
			for (size_t i = 0; i < totalbytes; ++i)
			{
				size_t elemi = i / nbytes;
				size_t outi = (elemi + 1) * nbytes - (i % nbytes);
				out[outi] = pb[i];
			}
			pb = out.c_str();
			TYPE_LOOKUP(_OUT_GENERIC, gencode)
		}
		else
		{
			TYPE_LOOKUP(_OUT_GENERIC, gencode)
		}
		return leaf;
	}

	/// Implementation of iLoader
	teq::TensptrT generate_func (std::string opname, teq::ArgsT args) override
	{
		if (args.empty())
		{
			logs::fatalf("cannot generate func %s without args", opname.c_str());
		}
		size_t gencode = egen::BAD_TYPE;
		auto arg = args[0].get_tensor().get();
		if (auto leaf = dynamic_cast<teq::iLeaf*>(arg))
		{
			gencode = leaf->type_code();
		}
		else if (auto func = dynamic_cast<teq::iOperableFunc*>(arg))
		{
			gencode = func->type_code();
		}
		else
		{
			logs::fatalf("cannot generate func from non-eteq tensor arg %s",
				arg->to_string().c_str());
		}
		teq::TensptrT func = nullptr;
		TYPE_LOOKUP(_OUT_GENFUNC, (egen::_GENERATED_DTYPE) gencode);
		return func;
	}

	/// Implementation of iLoader
	teq::CoordptrT generate_shaper (std::vector<double> coord) override
	{
		if (teq::mat_dim * teq::mat_dim != coord.size())
		{
			logs::fatal("cannot deserialize non-matrix shape map");
		}
		return std::make_shared<teq::CoordMap>(
			[&](teq::MatrixT fwd)
			{
				for (teq::RankT i = 0; i < teq::mat_dim; ++i)
				{
					for (teq::RankT j = 0; j < teq::mat_dim; ++j)
					{
						fwd[i][j] = coord[i * teq::mat_dim + j];
					}
				}
			});
	}

	/// Implementation of iLoader
	teq::CoordptrT generate_coorder (
		std::string opname, std::vector<double> coord) override
	{
		if (0 == coord.size()) // is identity
		{
			return nullptr;
		}
		if (teq::rank_cap + 1 < coord.size())
		{
			logs::fatal("cannot deserialize non-vector coordinate map");
		}
		bool is_bijective = false == estd::has(non_bijectives, egen::get_op(opname));
		teq::CoordT indices;
		auto cit = coord.begin();
		std::copy(cit, cit + teq::rank_cap, indices.begin());
		return std::make_shared<CoordMap>(indices, is_bijective);
	}
};

#undef _OUT_GENERIC

}

#endif // ETEQ_SERIALIZE_HPP
