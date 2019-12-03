#include <unordered_set>

#ifndef ETEQ_OBSERVABLE_HPP
#define ETEQ_OBSERVABLE_HPP

namespace eteq
{

template <typename SUB>
struct Observable
{
	virtual ~Observable (void) = default;

	void subscribe (SUB sub)
	{
		subs_.emplace(sub);
	}

	void unsubscribe (SUB sub)
	{
		subs_.erase(sub);
	}

protected:
	std::unordered_set<SUB> subs_;
};

}

#endif // ETEQ_OBSERVABLE_HPP
