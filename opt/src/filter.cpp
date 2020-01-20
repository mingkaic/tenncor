#include "opt/filter.hpp"

#ifdef OPT_RMDUPS_HPP

namespace opt
{

void remove_duplicates (teq::TensptrsT& roots, EqualF equals)
{
	teq::OwnerMapT owners = teq::track_owners(roots);
	teq::GraphStat stat;
	teq::ParentFinder pfinder;
	teq::TensMapT<std::vector<size_t>> rindices;
	for (size_t i = 0, n = roots.size(); i < n; ++i)
	{
		teq::TensptrT& root = roots[i];
		root->accept(stat);
		root->accept(pfinder);
		rindices[root.get()].push_back(i);
	}

	std::vector<teq::LeafptrT> csts;
	std::vector<teq::FuncptrT> functors;
	for (auto& gpair : stat.graphsize_)
	{
		auto tens = gpair.first;
		size_t height = gpair.second.upper_;
		if (0 == height)
		{
			auto leaf = std::static_pointer_cast<teq::iLeaf>(
				owners.at(tens).lock());
			if (teq::IMMUTABLE == leaf->get_usage())
			{
				csts.push_back(leaf);
			}
		}
		else
		{
			functors.push_back(std::static_pointer_cast<teq::iFunctor>(
				owners.at(tens).lock()));
		}
	}


	// remove equivalent nodes
	if (csts.size() > 0)
	{
		teq::debug("removing immutable duplicates");
		std::sort(csts.begin(), csts.end(),
			[](teq::LeafptrT a, teq::LeafptrT b)
			{
				return a->to_string() < b->to_string();
			});
		teq::LeafptrT cmp = csts[0];
		for (size_t i = 1, n = csts.size(); i < n; ++i)
		{
			if (equals(cmp, csts[i]))
			{
				// remove duplicate
				teq::debugf("replacing %s", csts[i]->to_string().c_str());
				// remove equivalent node
				replace_parents(pfinder, cmp, csts[i].get());
				auto it = rindices.find(csts[i].get());
				if (rindices.end() != it)
				{
					for (size_t ridx : it->second)
					{
						roots[ridx] = cmp;
					}
				}
			}
			else
			{
				cmp = csts[i];
			}
		}
	}

	if (functors.size() > 0)
	{
		teq::debug("removing functor duplicates");
		std::sort(functors.begin(), functors.end(),
			[&stat](teq::FuncptrT a, teq::FuncptrT b)
			{
				size_t lheight = stat.graphsize_[a.get()].upper_;
				size_t rheight = stat.graphsize_[b.get()].upper_;
				if (lheight == rheight)
				{
					return a->to_string() < b->to_string();
				}
				return lheight < rheight;
			});
		teq::FuncptrT cmp = functors[0];
		for (size_t i = 1, n = functors.size(); i < n; ++i)
		{
			if (equals(cmp, functors[i]))
			{
				teq::debugf("replacing functor %s",
					functors[i]->to_string().c_str());
				// remove equivalent node
				replace_parents(pfinder, cmp, functors[i].get());
				auto it = rindices.find(functors[i].get());
				if (rindices.end() != it)
				{
					for (size_t ridx : it->second)
					{
						roots[ridx] = cmp;
					}
				}
			}
			else
			{
				cmp = functors[i];
			}
		}
	}
}

teq::TensptrT constant_func (teq::FuncptrT& func,
	ParentReplF replace, CalcCvsF calc_func)
{
	auto children = func->get_children();
	if (std::all_of(children.begin(), children.end(),
		[&](teq::TensptrT ctens)
		{
			auto leaf = dynamic_cast<teq::iLeaf*>(ctens.get());
			return nullptr != leaf && teq::IMMUTABLE == leaf->get_usage();
		}))
	{
		teq::TensptrT converted = calc_func(func);
		if (func != converted)
		{
			replace(converted, func.get());
			return converted;
		}
	}
	return func;
}

void constant_funcs (teq::TensptrsT& roots, CalcCvsF calc_func)
{
	teq::OwnerMapT owners = teq::track_owners(roots);
	teq::GraphStat stat;
	teq::ParentFinder pfinder;
	teq::TensMapT<std::vector<size_t>> rindices;
	for (size_t i = 0, n = roots.size(); i < n; ++i)
	{
		teq::TensptrT& root = roots[i];
		root->accept(stat);
		root->accept(pfinder);
		rindices[root.get()].push_back(i);
	}

	teq::TensSetT constants;
	std::vector<teq::FuncptrT> functors;
	functors.reserve(stat.graphsize_.size());
	for (auto& gpair : stat.graphsize_)
	{
		if (gpair.second.upper_ > 0)
		{
			functors.push_back(std::static_pointer_cast<teq::iFunctor>(
				owners.at(gpair.first).lock()));
		}
		else if (teq::IMMUTABLE == static_cast<teq::iLeaf*>(
			gpair.first)->get_usage())
		{
			constants.emplace(gpair.first);
		}
	}
	std::sort(functors.begin(), functors.end(),
		[&stat](teq::FuncptrT a, teq::FuncptrT b)
		{
			return stat.graphsize_[a.get()].upper_ <
				stat.graphsize_[b.get()].upper_;
		});

	teq::TensptrSetT constant_roots;
	for (teq::FuncptrT& func : functors)
	{
		auto children = func->get_children();
		if (std::all_of(children.begin(), children.end(),
			[&](teq::TensptrT ctens)
			{
				return estd::has(constants, ctens.get()) ||
					estd::has(constant_roots, ctens);
			}))
		{
			// maintain functor constant roots
			for (teq::TensptrT child : children)
			{
				constant_roots.erase(child);
			}
			constants.emplace(func.get());
			constant_roots.emplace(func);
		}
	}
	// replace constant_roots funcs with their constant counter-part
	for (teq::TensptrT root : constant_roots)
	{
		teq::FuncptrT croot = std::static_pointer_cast<teq::iFunctor>(root);
		teq::TensptrT converted = calc_func(croot);
		if (croot != converted)
		{
			replace_parents(pfinder, converted, croot.get());
			auto it = rindices.find(croot.get());
			if (rindices.end() != it)
			{
				for (size_t ri : it->second)
				{
					roots[ri] = converted;
				}
			}
		}
	}
}

}

#endif
