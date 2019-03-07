///
/// serialize.hpp
/// ead
///
/// Purpose:
/// Define functions for marshal and unmarshal data sources
///

#include "pbm/data.hpp"

#include "ead/generated/opmap.hpp"

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

struct EADSaver : public pbm::iSaver
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
				for (uint8_t i = 0; i < ade::mat_dim; ++i)
				{
					for (uint8_t j = 0; j < ade::mat_dim; ++j)
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
		std::vector<double> out(coord.begin(), coord.end());
		out.push_back(static_cast<CoordMap*>(mapper.get())->transcode());
		return out;
	}
};

#define __OUT_GENERIC(realtype)leaf = is_const?\
ade::TensptrT(Constant<realtype>::get((realtype*) pb, shape)):\
ade::TensptrT(Variable<realtype>::get((realtype*) pb, shape, label));

/// Unmarshal cortenn::Source as Variable containing context of source
struct EADLoader : public pbm::iLoader
{
	ade::TensptrT generate_leaf (const char* pb, ade::Shape shape,
		size_t typecode, std::string label, bool is_const) override
	{
		ade::TensptrT leaf;
		age::_GENERATED_DTYPE gencode = (age::_GENERATED_DTYPE) typecode;
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
			TYPE_LOOKUP(__OUT_GENERIC, typecode)
		}
		else
		{
			TYPE_LOOKUP(__OUT_GENERIC, typecode)
		}
		return leaf;
	}

	ade::TensptrT generate_func (ade::Opcode opcode, ade::ArgsT args) override
	{
		return ade::TensptrT(ade::Functor::get(opcode, args));
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
				for (uint8_t i = 0; i < ade::mat_dim; ++i)
				{
					for (uint8_t j = 0; j < ade::mat_dim; ++j)
					{
						fwd[i][j] = coord[i * ade::mat_dim + j];
					}
				}
			});
	}

	ade::CoordptrT generate_coorder (
		ade::Opcode opcode, std::vector<double> coord) override
	{
		if (0 == coord.size()) // is identity
		{
			return nullptr;
		}
		if (ade::rank_cap + 1 != coord.size())
		{
			logs::fatal("cannot deserialize non-vector coordinate map");
		}
		bool is_bijective = non_bijectives.end() == non_bijectives.find(opcode.code_);
		ade::CoordT indices;
		auto cit = coord.begin();
		std::copy(cit, cit + ade::rank_cap, indices.begin());
		TransCode tcode = (TransCode) coord[ade::rank_cap];
		return std::make_shared<CoordMap>(tcode, indices, is_bijective);
	}
};

#undef __OUT_GENERIC

}

#endif // EAD_SERIALIZE_HPP
