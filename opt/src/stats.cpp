#include "opt/stats.hpp"

#ifdef OPT_STATS_HPP

namespace opt
{

bool is_scalar (teq::iLeaf* leaf)
{
	teq::Shape shape = leaf->shape();
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

std::string to_string (teq::CoordptrT c)
{
	if (teq::is_identity(c.get()))
	{
		return "";
	}
	return c->to_string();
}

bool lt (teq::CoordptrT a, teq::CoordptrT b)
{
	if (teq::is_identity(a.get()))
	{
		return false == teq::is_identity(b.get());
	}
	return a->to_string() < b->to_string();
}

bool is_equal (teq::CoordptrT a, teq::CoordptrT b)
{
	if (a == b)
	{
		return true;
	}
	if (teq::is_identity(a.get()) && teq::is_identity(b.get()))
	{
		return true;
	}
	if (nullptr != a && nullptr != b)
	{
		return a->to_string() == b->to_string();
	}
	return false;
}

bool lt (teq::TensSetT priorities,
	teq::iLeaf* a, teq::iLeaf* b)
{
	if (a == nullptr)
	{
		return true;
	}
	if (b == nullptr)
	{
		return false;
	}
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
			return estd::has(priorities, a);
		}
		std::hash<std::string> shash;
		return shash(ashape) < shash(bshape);
	}
	return atype < btype;
}

bool is_equal (teq::iLeaf* a, teq::iLeaf* b)
{
	teq::Shape shape = a->shape();
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

bool lt (teq::TensSetT priorities,
	teq::iFunctor* a, teq::iFunctor* b)
{
	if (a == nullptr)
	{
		return true;
	}
	if (b == nullptr)
	{
		return false;
	}
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
				if (tag::get_property_reg().has_property(a, tag::commutative_tag))
				{
					auto arg_lt =
					[](const teq::iFuncArg& a, const teq::iFuncArg& b)
					{
						auto atens = a.get_tensor().get();
						auto btens = b.get_tensor().get();
						if (atens != btens)
						{
							return atens < btens;
						}
						auto achild = static_cast<const teq::FuncArg*>(&a);
						auto bchild = static_cast<const teq::FuncArg*>(&b);
						return lt(ac->get_coorder(), bc->get_coorder());
					};
					std::sort(achildren.begin(), achildren.end(), arg_lt);
					std::sort(bchildren.begin(), bchildren.end(), arg_lt);
				}
				for (size_t i = 0; i < a_nchildren; ++i)
				{
					auto achild = static_cast<const teq::FuncArg*>(&achildren[i].get());
					auto bchild = static_cast<const teq::FuncArg*>(&bchildren[i].get());
					auto atens = achild->get_tensor().get();
					auto btens = bchild->get_tensor().get();
					if (atens != btens)
					{
						return atens < btens;
					}
					auto acoorder = achild->get_coorder();
					auto bcoorder = bchild->get_coorder();
					if (false == is_equal(acoorder, bcoorder))
					{
						return lt(acoorder, bcoorder);
					}
				}
				// a and b are equal, return true if a has priorities,
				return estd::has(priorities, a);
			}
			return a_nchildren < b_nchildren;
		}
		std::hash<std::string> shash;
		return shash(ashape) < shash(bshape);
	}
	return acode < bcode;
}

bool is_equal (teq::iFunctor* a, teq::iFunctor* b)
{
	if (a->get_opcode().code_ == b->get_opcode().code_)
	{
		teq::Shape shape = a->shape();
		if (shape.compatible_after(b->shape(), 0))
		{
			auto achildren = a->get_children();
			auto bchildren = b->get_children();
			// order doesn't matter, so normalize
			if (tag::get_property_reg().has_property(a, tag::commutative_tag))
			{
				auto arg_lt =
				[](teq::FuncArg a, teq::FuncArg b)
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
				[](const teq::iFuncArg& a, const teq::iFuncArg& b)
				{
					return a.get_tensor().get() == b.get_tensor().get() &&
						is_equal(
							static_cast<const teq::FuncArg*>(&a)->get_coorder(),
							static_cast<const teq::FuncArg*>(&b)->get_coorder());
				});
		}
	}
	return false;
}

}

#endif
