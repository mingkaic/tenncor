#include "opt/stats.hpp"

#ifdef OPT_STATS_HPP

namespace opt
{

bool is_scalar (ade::iLeaf* leaf)
{
	ade::Shape shape = leaf->shape();
	char* data = (char*) leaf->data();
	size_t n = shape.n_elems();
	size_t perbytes = leaf->nbytes() / n;
	for (size_t i = 1; i < n; ++i)
	{
		if (false == std::equal(data, data + perbytes,
			data + i * perbytes))
		{
			return false;
		}
	}
	return true;
}

std::string to_string (ade::CoordptrT c)
{
	if (ade::is_identity(c.get()))
	{
		return "";
	}
	return c->to_string();
}

bool lt (ade::CoordptrT a, ade::CoordptrT b)
{
	if (ade::is_identity(a.get()))
	{
		return false == ade::is_identity(b.get());
	}
	return a->to_string() < b->to_string();
}

bool is_equal (ade::CoordptrT a, ade::CoordptrT b)
{
	if (a == b)
	{
		return true;
	}
	if (ade::is_identity(a.get()) && ade::is_identity(b.get()))
	{
		return true;
	}
	if (nullptr != a && nullptr != b)
	{
		return a->to_string() == b->to_string();
	}
	return false;
}

bool lt (std::unordered_set<ade::iTensor*> priorities,
	ade::iLeaf* a, ade::iLeaf* b)
{
	size_t atype = a->type_code();
	size_t btype = b->type_code();
	if (atype == btype)
	{
		std::string ashape = a->shape().to_string();
		std::string bshape = b->shape().to_string();
		if (ashape == bshape)
		{
			// same shape and type = same nbytes
			size_t nbytes = a->nbytes();
			char* adata = (char*) a->data();
			char* bdata = (char*) b->data();
			for (size_t i = 0; i < nbytes; ++i)
			{
				if (adata[i] != bdata[i])
				{
					return adata[i] < bdata[i];
				}
			}
			// a and b are equal, return true if a has priorities,
			return util::has(priorities, a);
		}
		std::hash<std::string> shash;
		return shash(ashape) < shash(bshape);
	}
	return atype < btype;
}

bool is_equal (ade::iLeaf* a, ade::iLeaf* b)
{
	ade::Shape shape = a->shape();
	size_t dtype = a->type_code();
	if (shape.compatible_after(b->shape(), 0) &&
		dtype == b->type_code())
	{
		char* adata = (char*) a->data();
		char* bdata = (char*) b->data();
		size_t nbytes = a->nbytes();
		return std::equal(adata, adata + nbytes, bdata);
	}
	return false;
}

bool lt (std::unordered_set<ade::iTensor*> priorities,
	ade::iFunctor* a, ade::iFunctor* b)
{
	size_t acode = a->get_opcode().code_;
	size_t bcode = b->get_opcode().code_;
	if (acode == bcode)
	{
		std::string ashape = a->shape().to_string();
		std::string bshape = b->shape().to_string();
		if (ashape == bshape)
		{
			auto achildren = a->get_children();
			auto bchildren = b->get_children();
			size_t a_nchildren = achildren.size();
			size_t b_nchildren = bchildren.size();
			if (a_nchildren == b_nchildren)
			{
				// order doesn't matter, so normalize
				if (tag::has_property(a, tag::commutative_tag))
				{
					auto arg_lt =
					[](ade::FuncArg a, ade::FuncArg b)
					{
						auto atens = a.get_tensor().get();
						auto btens = b.get_tensor().get();
						if (atens != btens)
						{
							return atens < btens;
						}
						return lt(a.get_coorder(), b.get_coorder());
					};
					std::sort(achildren.begin(), achildren.end(), arg_lt);
					std::sort(bchildren.begin(), bchildren.end(), arg_lt);
				}
				for (size_t i = 0; i < a_nchildren; ++i)
				{
					auto atens = achildren[i].get_tensor().get();
					auto btens = bchildren[i].get_tensor().get();
					if (atens != btens)
					{
						return atens < btens;
					}
					auto acoorder = achildren[i].get_coorder();
					auto bcoorder = bchildren[i].get_coorder();
					if (false == is_equal(acoorder, bcoorder))
					{
						return lt(acoorder, bcoorder);
					}
				}
				// a and b are equal, return true if a has priorities,
				return util::has(priorities, a);
			}
			return a_nchildren < b_nchildren;
		}
		std::hash<std::string> shash;
		return shash(ashape) < shash(bshape);
	}
	return acode < bcode;
}

bool is_equal (ade::iFunctor* a, ade::iFunctor* b)
{
	if (a->get_opcode().code_ == b->get_opcode().code_)
	{
		ade::Shape shape = a->shape();
		if (shape.compatible_after(b->shape(), 0))
		{
			auto achildren = a->get_children();
			auto bchildren = b->get_children();
			if (tag::has_property(a, tag::commutative_tag)) // order doesn't matter, so normalize
			{
				auto arg_lt =
				[](ade::FuncArg a, ade::FuncArg b)
				{
					auto atens = a.get_tensor().get();
					auto btens = b.get_tensor().get();
					if (atens != btens)
					{
						return atens < btens;
					}
					return lt(a.get_coorder(), b.get_coorder());
				};
				std::sort(achildren.begin(), achildren.end(), arg_lt);
				std::sort(bchildren.begin(), bchildren.end(), arg_lt);
			}
			return std::equal(achildren.begin(), achildren.end(),
				bchildren.begin(),
				[](const ade::FuncArg& a, const ade::FuncArg& b)
				{
					return a.get_tensor().get() == b.get_tensor().get() &&
						is_equal(a.get_coorder(), b.get_coorder());
				});
		}
	}
	return false;
}

}

#endif