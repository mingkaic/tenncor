
#include "eteq/eteq.hpp"

#ifndef DBG_GRAPHEQ_HPP
#define DBG_GRAPHEQ_HPP

/// Return true if lroot and rroot graphs are structurally equal
template <typename T>
bool is_equal (const eteq::ETensor<T>& lroot, const eteq::ETensor<T>& rroot)
{
	if (lroot.get() == rroot.get())
	{
		return true;
	}

	teq::GraphIndex lindex;
	teq::GraphIndex rindex;
	lroot->accept(lindex);
	rroot->accept(rindex);
	size_t graphsize = lindex.indices_.size();
	if (graphsize != rindex.indices_.size())
	{
		return false;
	}

	std::vector<std::pair<teq::iTensor*,teq::iTensor*>> orders(graphsize);
	for (auto& lpair : lindex.indices_)
	{
		orders[lpair.second].first = lpair.first;
	}
	for (auto& rpair : rindex.indices_)
	{
		orders[rpair.second].second = rpair.first;
	}

	teq::GraphStat lstat;
	teq::GraphStat rstat;
	lroot->accept(lstat);
	rroot->accept(rstat);

	eteq::Hasher<T> hasher;
	for (auto& ords : orders)
	{
		auto& lheights = lstat.at(ords.first);
		auto& rheights = rstat.at(ords.second);
		if (lheights.upper_ != rheights.upper_ ||
			lheights.lower_ != rheights.lower_)
		{
			return false;
		}
		if (lheights.upper_ == 0)
		{
			auto hid = hasher.uuid_gen_();
			hasher.hashes_.emplace(ords.first, hid);
			hasher.hashes_.emplace(ords.second, hid);
			hasher.visited_.emplace(ords.first);
			hasher.visited_.emplace(ords.second);
		}
	}
	lroot->accept(hasher);
	rroot->accept(hasher);

	return hasher.at(lroot.get()) == hasher.at(rroot.get());
}

/// Return true if lroot and rroot graphs have the same data
template <typename T>
bool is_dataeq (const eteq::ETensor<T>& lroot, const eteq::ETensor<T>& rroot)
{
	if (lroot.get() == rroot.get())
	{
		return true;
	}

	teq::GraphIndex lindex;
	teq::GraphIndex rindex;
	lroot->accept(lindex);
	rroot->accept(rindex);
	size_t graphsize = lindex.indices_.size();
	if (graphsize != rindex.indices_.size())
	{
		return false;
	}

	std::vector<std::pair<teq::iTensor*,teq::iTensor*>> orders(graphsize);
	for (auto& lpair : lindex.indices_)
	{
		orders[lpair.second].first = lpair.first;
	}
	for (auto& rpair : rindex.indices_)
	{
		orders[rpair.second].second = rpair.first;
	}

	return std::all_of(orders.begin(), orders.end(),
		[](const std::pair<teq::iTensor*,teq::iTensor*>& ord)
		{
			teq::Shape shape = ord.first->shape();
			if (false == std::equal(shape.begin(), shape.end(),
				ord.second->shape().begin()))
			{
				return false;
			}
			const T* lptr = (const T*) ord.first->device().data();
			const T* rptr = (const T*) ord.second->device().data();
			return std::equal(lptr, lptr + shape.n_elems(), rptr);
		});
}

/// Return percent of nodes that are data equivalent
template <typename T>
double percent_dataeq (const eteq::ETensor<T>& lroot, const eteq::ETensor<T>& rroot)
{
	if (lroot.get() == rroot.get())
	{
		return 1.;
	}

	teq::GraphIndex lindex;
	teq::GraphIndex rindex;
	lroot->accept(lindex);
	rroot->accept(rindex);
	size_t graphsize = lindex.indices_.size();
	if (graphsize != rindex.indices_.size())
	{
		return 0.;
	}

	std::vector<std::pair<teq::iTensor*,teq::iTensor*>> orders(graphsize);
	for (auto& lpair : lindex.indices_)
	{
		orders[lpair.second].first = lpair.first;
	}
	for (auto& rpair : rindex.indices_)
	{
		orders[rpair.second].second = rpair.first;
	}

	size_t nequals = 0;
	for (const std::pair<teq::iTensor*,teq::iTensor*>& ord : orders)
	{
		teq::Shape shape = ord.first->shape();
		const T* lptr = (const T*) ord.first->device().data();
		const T* rptr = (const T*) ord.second->device().data();
		if (std::equal(shape.begin(), shape.end(),
				ord.second->shape().begin()) &&
			std::equal(lptr, lptr + shape.n_elems(), rptr))
		{
			++nequals;
		}
	}
	return nequals / (double) orders.size();
}

#endif // DBG_GRAPHEQ_HPP
