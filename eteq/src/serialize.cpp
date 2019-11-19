#include "eteq/serialize.hpp"

#ifdef ETEQ_SERIALIZE_HPP

namespace eteq
{

static bool is_big_endian(void)
{
	union
	{
		uint16_t _;
		char bytes[2];
	} twob = { 0x0001 };

	return twob.bytes[0] == 0;
}

static std::string save_leaf (teq::iLeaf* leaf)
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

pbm::TensMapIndicesT save_graph (
	tenncor::Graph& out, teq::TensptrsT roots,
	tag::TagRegistry& registry)
{
	return pbm::save_graph(out, roots, registry, save_leaf);
}

#define _OUT_GENERIC(realtype)leaf = is_const?\
teq::TensptrT(Constant<realtype>::get((realtype*) pb, shape)):\
teq::TensptrT(Variable<realtype>::get((realtype*) pb, shape, label));

static teq::TensptrT load_leaf (
	const tenncor::Source& source, std::string label)
{
	teq::Shape shape = pbm::get_shape(source);
	const char* pb = source.data().c_str();
	bool is_const = source.is_const();

	teq::TensptrT leaf;
	egen::_GENERATED_DTYPE gencode = egen::get_type(source.typelabel());
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

#undef _OUT_GENERIC

#define _OUT_GENFUNC(realtype)func = convert_func<realtype>(opname, edges);

static teq::TensptrT load_func (std::string opname, const pbm::EdgesT& edges)
{
	if (edges.empty())
	{
		logs::fatalf("cannot generate func %s without edges", opname.c_str());
	}
	size_t gencode = egen::BAD_TYPE;
	auto ctens = edges[0].first.get();
	if (auto leaf = dynamic_cast<teq::iLeaf*>(ctens))
	{
		gencode = leaf->type_code();
	}
	else if (auto func = dynamic_cast<teq::iOperableFunc*>(ctens))
	{
		gencode = func->type_code();
	}
	else
	{
		logs::fatalf("cannot generate func from non-eteq tensor arg %s",
			ctens->to_string().c_str());
	}
	teq::TensptrT func = nullptr;
	TYPE_LOOKUP(_OUT_GENFUNC, (egen::_GENERATED_DTYPE) gencode);
	return func;
}

#undef _OUT_GENFUNC

void load_graph (teq::TensptrSetT& roots,
	const tenncor::Graph& pb_graph,
	tag::TagRegistry& registry)
{
	pbm::load_graph(roots, pb_graph, registry, load_leaf, load_func);
}

}

#endif
