#include "ade/ade.hpp"

#include "ead/opt/nodes.hpp"

#ifndef EAD_OPS_REUSE_HPP
#define EAD_OPS_REUSE_HPP

namespace ead
{

static std::unordered_set<size_t> communative_codes =
{
	age::ADD,
	age::MUL,
	age::MIN,
	age::MAX,
	age::EQ,
	age::NEQ,
};

template <typename T>
static bool const_equals (Constant<T>* lhs, Constant<T>* rhs)
{
	auto& lhs_shape = lhs->shape();
	auto& rhs_shape = rhs->shape();
	if (false == std::equal(lhs_shape.begin(),
		lhs_shape.end(), rhs_shape.begin()))
	{
		return false;
	}

	char* lhs_ptr = (char*) lhs->data();
	char* rhs_ptr = (char*) rhs->data();
	return std::equal(lhs_ptr, lhs_ptr +
		lhs_shape.n_elems() * sizeof(T), rhs_ptr);
}

static bool coorder_equal (ade::CoordptrT lhs, ade::CoordptrT rhs)
{
	if (lhs == rhs) // if both are null, then return true
	{
		return true;
	}
	else if (lhs == nullptr || rhs == nullptr)
	{
		// one is null, other is not
		return false;
	}
	ade::CoordT lhsc;
	ade::CoordT rhsc;
	lhs->forward(lhsc.begin(), lhsc.begin());
	rhs->forward(rhsc.begin(), rhsc.begin());
	return std::equal(lhsc.begin(), lhsc.end(), rhsc.begin());
}

static bool func_equals (ade::iFunctor* lhs, ade::iFunctor* rhs,
	const std::unordered_map<ade::iTensor*,ade::iTensor*>& orig2new)
{
	auto& lhs_shape = lhs->shape();
	auto& rhs_shape = rhs->shape();
	if (false == std::equal(lhs_shape.begin(),
		lhs_shape.end(), rhs_shape.begin()))
	{
		return false;
	}

	auto et = orig2new.end();
	auto& lhs_children = lhs->get_children();
	auto& rhs_children = rhs->get_children();
	auto arg_equal =
		[&orig2new,&et](
			const ade::FuncArg& lhs,
			const ade::FuncArg& rhs)
		{
			auto lhs_tens = lhs.get_tensor().get();
			auto rhs_tens = rhs.get_tensor().get();
			auto lit = orig2new.find(lhs_tens);
			auto rit = orig2new.find(rhs_tens);
			if (lit != et)
			{
				lhs_tens = lit->second;
			}
			if (rit != et)
			{
				rhs_tens = rit->second;
			}
			return lhs_tens == rhs_tens &&
				coorder_equal(lhs.get_coorder(), rhs.get_coorder());
		};
	if (communative_codes.end() == communative_codes.find(
		(age::_GENERATED_OPCODE) lhs->get_opcode().code_))
	{
		// order matters
		// child check is expensive so check size before equality
		return std::equal(lhs_children.begin(), lhs_children.end(),
			rhs_children.begin(), arg_equal);
	}
	size_t nchildren = lhs_children.size();
	if (nchildren != rhs_children.size())
	{
		return false;
	}
	auto rit = rhs_children.begin();
	auto ret = rhs_children.end();
	return std::all_of(lhs_children.begin(), lhs_children.end(),
		[&](const ade::FuncArg& lhs_child)
		{
			return std::any_of(rit, ret,
				[&](const ade::FuncArg& rhs_child)
				{
					return arg_equal(lhs_child, rhs_child);
				});
		});
}

template <typename T>
NodesT<T> ops_reuse (NodesT<T> roots)
{
	std::unordered_map<ade::iTensor*,NodeptrT<T>> smart_map;
	ade::GraphStat stat;
	for (NodeptrT<T>& root : roots)
	{
		auto tens = root->get_tensor();
		smart_map.emplace(tens.get(), root);
		tens->accept(stat);
	}
	if (stat.graphsize_.size() == 0)
	{
		return roots;
	}

	size_t max_graphsize = 0;
	for (NodeptrT<T>& root : roots)
	{
		max_graphsize = std::max(max_graphsize,
			stat.graphsize_[root->get_tensor().get()].upper_ + 1);
	}

	std::vector<std::list<ade::iTensor*>> tens(max_graphsize);
	for (std::pair<ade::iTensor*,ade::NumRange<size_t>> graphpair :
		stat.graphsize_)
	{
		ade::iTensor* ten = graphpair.first;
		size_t index = graphpair.second.upper_;
		// is root
		if (smart_map.end() != smart_map.find(ten))
		{
			// always push roots to the front
			tens[index].push_front(ten);
		}
		else
		{
			tens[index].push_back(ten);
		}
	}

	// assert stat.graphsize_.size() > 0, hence tens.size() > 0
	std::unordered_map<ade::iTensor*,ade::iTensor*> orig2new;
	{
		std::unordered_map<size_t,std::list<Constant<T>*>> hashs;
		for (ade::iTensor* leaf : tens[0])
		{
			auto lfs = static_cast<iLeaf<T>*>(leaf);
			if (lfs->is_const())
			{
				auto cst = static_cast<Constant<T>*>(lfs);
				bool not_found = true;
				auto& shape = cst->shape();
				size_t hashidx = std::hash<std::string>()(
					std::string(shape.begin(), shape.end()));
				auto& potential_eqs = hashs[hashidx];
				for (Constant<T>* potential_eq : potential_eqs)
				{
					if (const_equals(cst, potential_eq))
					{
						orig2new.emplace(cst, potential_eq);
						not_found = false;
						break;
					}
				}
				if (not_found)
				{
					potential_eqs.push_back(cst);
				}
			}
		}
	}

	for (size_t i = 1, n = tens.size(); i < n; ++i)
	{
		std::unordered_map<size_t,std::list<ade::iFunctor*>> hashs;
		for (ade::iTensor* ten : tens[i])
		{
			bool not_found = true;
			auto func = static_cast<ade::iFunctor*>(ten);
			// populate smart map
			auto& children = func->get_children();
			for (auto& child : children)
			{
				auto smart = child.get_tensor();
				smart_map[smart.get()] = to_node<T>(smart);
			}

			// find equalities
			size_t hashidx = func->get_opcode().code_;
			auto& potential_eqs = hashs[hashidx];
			for (ade::iFunctor* potential_eq : potential_eqs)
			{
				if (func_equals(func, potential_eq, orig2new))
				{
					orig2new.emplace(func, potential_eq);
					not_found = false;
					break;
				}
			}
			if (not_found)
			{
				potential_eqs.push_back(func);
			}
		}
	}
	std::unordered_map<ade::iTensor*,std::vector<ade::iTensor*>> new2origs;
	for (auto& replace_pair : orig2new)
	{
		new2origs[replace_pair.second].push_back(replace_pair.first);
	}

	for (size_t i = 1, n = tens.size(); i < n; ++i)
	{
		for (ade::iTensor* ten : tens[i])
		{
			// only update functors that are not replaced
			if (orig2new.end() == orig2new.find(ten))
			{
				auto func = static_cast<ade::iFunctor*>(ten);
				bool changed = false;
				ArgsT<T> children = ade_to_ead_args<T>(func->get_children());
				for (size_t i = 0, n = children.size(); i < n; ++i)
				{
					auto it = orig2new.find(children[i].get_tensor().get());
					if (orig2new.end() != it)
					{
						changed = true;
						children[i] = FuncArg<T>
						{
							smart_map[it->second],
							children[i].get_shaper(),
							children[i].get_coorder(),
						};
					}
				}
				if (changed)
				{
					auto f = Functor<T>::get(func->get_opcode(), children);
					auto optimized = to_node<T>(ade::TensptrT(f));
					// update smart and orig2new
					smart_map.emplace(f, optimized);
					auto it = new2origs.find(func);
					if (new2origs.end() != it)
					{
						// reference new updated functor instead of old one
						for (ade::iTensor* orig : it->second)
						{
							orig2new[orig] = f;
						}
					}
					orig2new[func] = f;
					new2origs[f].push_back(func);
				}
			}
		}
	}

	for (auto& root : roots)
	{
		auto rit = orig2new.find(root->get_tensor().get());
		if (orig2new.end() != rit)
		{
			root = smart_map[rit->second];
		}
	}
	return roots;
}

}

#endif // EAD_OPS_REUSE_HPP
