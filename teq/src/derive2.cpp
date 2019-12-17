#include <cassert>
#include <queue>

#include "teq/derive.hpp"

#ifdef TEQ_DERIVE2_HPP

namespace teq
{

struct Grader : public OnceTraveler
{
	Grader (OwnerMapT owners, TensCIdxT roadmap,
		iDerivativeFuncs& dfuncs) :
		owners_(owners), roadmap_(roadmap), dfuncs_(dfuncs) {}

	// Maps u to dF(x) / du, where F(x) is the root_
	std::unordered_map<const iTensor*,TensptrsT> grads_;

	std::queue<iTensor*> nexts_;

private:
	void visit_leaf (iLeaf& leaf) override {}

	void visit_func (iFunctor& func) override
	{
		if (false == estd::has(grads_, &func))
		{
			grads_.emplace(&func, TensptrsT{
				dfuncs_.get_const_one(func.shape())});
		}
		TensptrsT prevs = estd::must_getf(grads_, &func,
			"failed to find derivative with respect to %s",
			func.to_string().c_str());
		assert(prevs.size() > 0);
		TensptrT bwd = prevs.size() == 1 ? prevs.front() : dfuncs_.add(prevs);
		auto& nexts = roadmap_[&func];
		// for each painted child, calculate dThis/dChild
		// go through grads in order
		auto fptr = std::static_pointer_cast<iFunctor>(
			owners_.at(&func).lock());
		auto children = func.get_children();
		size_t nchildren = children.size();
		for (size_t i : nexts)
		{
			assert(i < nchildren);
			auto ctens = children[i].get();
			auto local = dfuncs_.local_derivative(fptr, i);
			auto grad_step = dfuncs_.chain_rule(fptr, local, bwd, i);
			grads_[ctens].push_back(grad_step);
			nexts_.push(ctens);
		}
	}

	OwnerMapT owners_;

	TensCIdxT roadmap_;

	iDerivativeFuncs& dfuncs_;
};

TensptrT derive (TensptrT root, TensptrT target, iDerivativeFuncs& funcs)
{
	if (target == nullptr)
	{
		return funcs.get_const_zero(Shape());
	}

	if (root == nullptr)
	{
		return funcs.get_const_zero(target->shape());
	}

	if (root == target)
	{
		return funcs.get_const_one(target->shape());
	}

	PathFinder finder(target.get());
	root->accept(finder);

	auto& roadmap = finder.roadmap_;
	// no path to wrt
	if (roadmap.empty())
	{
		return funcs.get_const_zero(target->shape());
	}
	// else there exists a path to wrt
	// using pathfinder, breadth first traverse from this to wrt
	auto owners = track_owners({root});

	Grader grader(owners, roadmap, funcs);
	root->accept(grader);
	while (false == grader.nexts_.empty())
	{
		grader.nexts_.front()->accept(grader);
		grader.nexts_.pop();
	}

	TensptrsT tgrads = estd::must_getf(grader.grads_, target.get(),
		"failed to find derivative with respect to %s",
		target->to_string().c_str());
	assert(tgrads.size() > 0);
	return tgrads.size() == 1 ? tgrads.front() : funcs.add(tgrads);
}

}

#endif
