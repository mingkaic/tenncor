
#include "dbg/compare/equal.hpp"

#ifdef DBG_GRAPHEQ_HPP

bool is_equal (const eteq::ETensor& lroot, const eteq::ETensor& rroot)
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

	eteq::Hasher hasher;
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
			auto hid = global::get_uuidengine()();
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

bool is_dataeq (const eteq::ETensor& lroot, const eteq::ETensor& rroot)
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
			auto dtype = (egen::_GENERATED_DTYPE) ord.first->get_meta().type_code();
			if (false == std::equal(shape.begin(), shape.end(),
				ord.second->shape().begin()) ||
				dtype != ord.second->get_meta().type_code())
			{
				return false;
			}
			const char* lptr = (const char*) ord.first->device().data();
			const char* rptr = (const char*) ord.second->device().data();
			return std::equal(lptr,
				lptr + shape.n_elems() * egen::type_size(dtype), rptr);
		});
}

double percent_dataeq (const eteq::ETensor& lroot, const eteq::ETensor& rroot)
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
		auto dtype = (egen::_GENERATED_DTYPE) ord.first->get_meta().type_code();
		if (false == std::equal(shape.begin(), shape.end(),
			ord.second->shape().begin()) ||
			dtype != ord.second->get_meta().type_code())
		{
			continue;
		}
		const char* lptr = (const char*) ord.first->device().data();
		const char* rptr = (const char*) ord.second->device().data();
		if (std::equal(lptr,
			lptr + shape.n_elems() * egen::type_size(dtype), rptr))
		{
			++nequals;
		}
	}
	return nequals / (double) orders.size();
}

#endif
