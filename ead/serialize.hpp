///
/// serialize.hpp
/// ead
///
/// Purpose:
/// Define functions for marshal and unmarshal data sources
///

#include "pbm/data.hpp"

#include "ead/generated/opcode.hpp"
#include "ead/generated/dtype.hpp"

#include "ead/coord.hpp"
#include "ead/constant.hpp"
#include "ead/variable.hpp"

#ifndef EAD_SERIALIZE_HPP
#define EAD_SERIALIZE_HPP

namespace ead
{

static std::unordered_set<size_t> non_bijectives =
{
	age::REDUCE_SUM,
	age::REDUCE_PROD,
	age::REDUCE_MIN,
	age::REDUCE_MAX,
	age::EXTEND,
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

struct EADSaver final : public pbm::iSaver
{
	std::string save_leaf (bool& is_const, ade::iLeaf* leaf) override
	{
		char* data = (char*) leaf->data();
		size_t nelems = leaf->shape().n_elems();
		size_t nbytes = age::type_size((age::_GENERATED_DTYPE) leaf->type_code());
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

	std::vector<double> save_shaper (const ade::CoordptrT& mapper) override
	{
		std::vector<double> out;
		mapper->access(
			[&out](const ade::MatrixT& mat)
			{
				for (ade::RankT i = 0; i < ade::mat_dim; ++i)
				{
					for (ade::RankT j = 0; j < ade::mat_dim; ++j)
					{
						out.push_back(mat[i][j]);
					}
				}
			});
		return out;
	}

	std::vector<double> save_coorder (const ade::CoordptrT& mapper) override
	{
		if (nullptr == mapper)
		{
			return std::vector<double>();
		}
		ade::CoordT coord;
		mapper->forward(coord.begin(), coord.begin());
		return std::vector<double>(coord.begin(), coord.end());
	}
};

#define _OUT_GENERIC(realtype)leaf = is_const?\
make_constant<realtype>((realtype*) pb, shape)->get_tensor():\
ade::TensptrT(Variable<realtype>::get((realtype*) pb, shape, label));

/// Unmarshal cortenn::Source as Variable containing context of source
struct EADLoader final : public pbm::iLoader
{
	ade::TensptrT generate_leaf (const char* pb, ade::Shape shape,
		std::string typelabel, std::string label, bool is_const) override
	{
		ade::TensptrT leaf;
		age::_GENERATED_DTYPE gencode = age::get_type(typelabel);
		size_t nbytes = age::type_size(gencode);
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

	ade::TensptrT generate_func (std::string opname, ade::ArgsT args) override
	{
		return ade::TensptrT(ade::Functor::get(ade::Opcode{opname, age::get_op(opname)}, args));
	}

	ade::CoordptrT generate_shaper (std::vector<double> coord) override
	{
		if (ade::mat_dim * ade::mat_dim != coord.size())
		{
			logs::fatal("cannot deserialize non-matrix shape map");
		}
		return std::make_shared<ade::CoordMap>(
			[&](ade::MatrixT fwd)
			{
				for (ade::RankT i = 0; i < ade::mat_dim; ++i)
				{
					for (ade::RankT j = 0; j < ade::mat_dim; ++j)
					{
						fwd[i][j] = coord[i * ade::mat_dim + j];
					}
				}
			});
	}

	ade::CoordptrT generate_coorder (
		std::string opname, std::vector<double> coord) override
	{
		if (0 == coord.size()) // is identity
		{
			return nullptr;
		}
		if (ade::rank_cap + 1 < coord.size())
		{
			logs::fatal("cannot deserialize non-vector coordinate map");
		}
		bool is_bijective = false == estd::has(non_bijectives, age::get_op(opname));
		ade::CoordT indices;
		auto cit = coord.begin();
		std::copy(cit, cit + ade::rank_cap, indices.begin());
		return std::make_shared<CoordMap>(indices, is_bijective);
	}
};

#undef _OUT_GENERIC

}

#endif // EAD_SERIALIZE_HPP
