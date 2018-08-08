#include "glass/session.hpp"

#ifdef GLASS_SESSION_HPP

void Session::add (std::string label, Nodeptr& node)
{
	iNode* nptr = node.get();
	auto nit = nodes_.find(label);
	auto lit = names_.find(nptr);
	if (nodes_.end() != nit ||
		names_.end() != lit)
	{
		handle_error("conflicting node",
			ErrArg<std::string>("name", label),
			ErrArg<void*>("pointer", nptr));
	}
	nodes_[label] = node.ref();
	names_[nptr] = label;
}

std::string Session::name (Nodeptr& node) const
{
	iNode* nptr = node.get();
	auto it = names_.find(nptr);
	if (names_.end() != it)
	{
		handle_error("no such node",
			ErrArg<void*>("pointer", nptr));
	}
	return it->second;
}

iNode* Session::get (std::string key) const
{
	auto it = nodes_.find(key);
	if (nodes_.end() == it)
	{
		handle_error("no such node",
			ErrArg<std::string>("name", key));
	}
	auto ref = it->second;
	if (ref.expired())
	{
		handle_error("no such node",
			ErrArg<std::string>("name", key));
	}
	return ref.lock().get();
}

void Session::clean (void)
{
	NamedNodes ncpys = nodes_;
	for (auto npairs : ncpys)
	{
		if (npairs.second.expired())
		{
			nodes_.erase(npairs.first);
		}
	}
}

std::string Session::hash (void) const
{
	return uid_;
}

#endif
