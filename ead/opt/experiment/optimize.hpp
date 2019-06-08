#include <list>

#include "ade/ade.hpp"

#include "tag/prop.hpp"

#include "ead/constant.hpp" // to get immutable_tag
#include "ead/helper.hpp" // to get commutative_tag

bool is_immutable (const ade::iLeaf*& leaf)
{
	auto reps = tag::get_tags(leaf);
	auto it = reps.find(tag::props_key);
	if (reps.end() == it)
	{
		return false;
	}
	return it->second.end() != std::find(
		it->second.begin(), it->second.end(), ead::immutable_tag);
}

bool is_commutative (const ade::iFunctor*& func)
{
	auto reps = tag::get_tags(func);
	auto it = reps.find(tag::props_key);
	if (reps.end() == it)
	{
		return false;
	}
	return it->second.end() != std::find(
		it->second.begin(), it->second.end(), ead::commutative_tag);
}

bool lt (const ade::FuncArg& a, const ade::FuncArg& b)
{
	auto atens = a.get_tensor().get();
	auto btens = b.get_tensor().get();
	if (atens == btens)
	{
		return std::hash(a.get_coorder()->to_string()) <
			std::hash(b.get_coorder()->to_string());
	}
	return atens < btens;
}

bool lt (const ade::iLeaf*& a, const ade::iLeaf*& b)
{
	size_t atype = a->type_code();
	size_t btype = b->type_code();
	if (atype == btype)
	{
		std::string ashape = a->shape().to_string();
		std::string bshape = b->shape().to_string();
		if (ashape == bshape)
		{
			char* adata = (char*) a->data();
			char* bdata = (char*) b->data();
			size_t nbytes = shape.n_elems() *
				age::type_size((age::_GENERATED_DTYPE) dtype);
			return std::equal(adata, adata + nbytes, bdata);
		}
		return std::hash(ashape) < std::hash(bshape);
	}
	return atype < btype;
}

bool is_equal (const ade::FuncArg& a, const ade::FuncArg& b)
{
	if (a.get_tensor().get() == b.get_tensor().get())
	{
		return a.get_coorder()->to_string() ==
			b.get_coorder()->to_string();
	}
	return false;
}

// for any ileaf pair a-b, they are equivalent IFF they are both tagged immutable AND
// share same shape and data values
bool is_equal (const ade::iLeaf*& a, const ade::iLeaf*& b)
{
	ade::Shape shape = a->shape();
	size_t dtype = a->type_code();
	if (shape.compatible_after(b->shape(), 0) &&
		dtype == b->type_code())
	{
		char* adata = (char*) a->data();
		char* bdata = (char*) b->data();
		size_t nbytes = shape.n_elems() *
			age::type_size((age::_GENERATED_DTYPE) dtype);
		return std::equal(adata, adata + nbytes, bdata);
	}
	return false;
}

// for any functors a-b, they are equivalent IFF a and b are the same opcode AND
// share identical function arguments (same children, shapers, and coorders)
// order matters UNLESS the op is tagged as commutative
bool is_equal (const ade::iFunctor*& a, const ade::iFunctor*& b)
{
	if (a->get_opcode().code_ == b->get_opcode())
	{
		ade::Shape shape = a->shape();
		if (shape.compatible_after(b->shape(), 0))
		{
			auto achildren = a->get_children();
			auto bchildren = b->get_children();
			if (is_commutative(a)) // order doesn't matter, so normalize
			{
				std::sort(achildren.begin(), achildren.end(), lt);
				std::sort(bchildren.begin(), bchildren.end(), lt);
			}
			return std::equal(achildren.begin(), achildren.end(),
				bchildren.begin(), is_equal);
		}
	}
	return false;
}

void optimize (ade::TensT roots)
{
	if (roots.empty())
	{
		return;
	}

	ade::GraphStat stat;
	ade::ParentFinder pfinder;
	ade::OwnerMapT owners = ade::track_owners(roots);
	for (ade::TensptrT& root : roots)
	{
		root->accept(stat);
		root->accept(pfinder);
	}

	// stat provides positional information:
	//		- nodes of different height will never be equivalent
	// pfinder provides adjacency information:
	//		- parents of equivalent/converted nodes will need updating
	// for each height from 0 to max:
	//		assert: every node below height is optimal and unique (non-equivalent from each other)
	//		1. delete and update equivalent nodes on the same height
	//		2. for each node at height level,
	//			apply rule conversion to non-equivalent generate converted subgraph
	//		3. delete and update equivalent nodes in converted subgraph

	std::vector<size_t> root_heights;
	root_heights.reserve(roots.size());
	std::transform(roots.begin(), roots.end(),
		std::back_inserter(root_heights),
		[&stat](ade::TensptrT& root)
		{
			return stat.graphsize_[root.get()].upper_;
		});
	// max of the maxheight of roots should be the maxheight of the whole graph
	size_t maxheight = *std::max_element(
		root_heights.begin(), root_heights.end());

	std::vector<ade::iLeaf*> leaves;
	std::vector<std::vector<ade::iFunctor*>> functors(maxheight - 1);

	for (auto& gpair : stat.graphsize_)
	{
		auto tens = gpair.first;
		size_t height = gpair.second.upper_;
		if (0 == height)
		{
			leaves.push_back(static_cast<ade::iLeaf*>(tens));
		}
		else
		{
			functors[height - 1].push_back(static_cast<ade::iFunctor*>(tens));
		}
	}

	// there are no conversions for leaves
	// remove equivalent nodes
	std::vector<ade::iLeaf*> immutables;
	immutables.reserve(leaves.size());
	std::copy_if(leaves.begin(), leaves.end(),
		immutables.begin(), is_immutable);
	if (false == immutables.empty())
	{
		std::sort(immutables.begin(), immutables.end(), lt);
		ade::iLeaf* last = immutables[0];
		for (size_t i = 1, n = immutables.size(); i < n; ++i)
		{
			auto cur = immutables[i];
			if (is_equal(last, cur))
			{
				// remove equivalent node
				auto& parents = pfinder.parents_[cur];
				for (auto& parent_pair : parents)
				{
					auto f = static_cast<ade::iFunctor*>(parent_pair.first);
					auto& children = f->get_children();
					for (size_t j : parent_pair.second)
					{
						ade::FuncArg arg(owners(last),
							children[j].get_shaper(),
							children[j].map_io(),
							children[j].get_coorder());
						f->update_child(arg, j);
					}
				}
			}
			else
			{
				last = cur;
			}
		}
	}

	for (size_t i = 1; i < maxheight; ++i)
	{
		std::vector<ade::iFunctor*>& funcs = functors[i - 1];
		// remove equivalent nodes
		if (funcs.empty())
		{
			// todo: warn
			continue;
		}
		std::sort(funcs.begin(), funcs.end(), lt);
		ade::iFunctor* last = funcs[0];
		for (size_t i = 1, n = funcs.size(); i < n; ++i)
		{
			auto cur = funcs[i];
			if (is_equal(last, cur))
			{
				// remove equivalent node
				auto& parents = pfinder.parents_[cur];
				for (auto& parent_pair : parents)
				{
					auto f = static_cast<ade::iFunctor*>(parent_pair.first);
					auto& children = f->get_children();
					for (size_t j : parent_pair.second)
					{
						ade::FuncArg arg(owners(last),
							children[j].get_shaper(),
							children[j].map_io(),
							children[j].get_coorder());
						f->update_child(arg, j);
					}
				}
			}
			else
			{
				last = cur;
			}
		}

		// apply rule conversion to uniques

		// remove equivalent nodes in converted subgraph
	}
}
