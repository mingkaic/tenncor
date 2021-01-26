///
/// once.hpp
/// teq
///
/// Purpose:
/// Define holder object that calls signal once upon destruction
///

#ifndef TEQ_ONCE_HPP
#define TEQ_ONCE_HPP

#include "jobs/jobs.hpp"

namespace teq
{

template <typename T>
struct Once final : public jobs::ScopeGuard
{
	Once (T obj, jobs::GuardOpF killsig = jobs::GuardOpF()) :
		ScopeGuard(killsig), obj_(obj) {}

	template <typename OT>
	Once (T altobj, Once<OT>&& other) :
		ScopeGuard(std::move(other)), obj_(altobj) {}

	Once (const Once<T>& other) = delete;

	Once (Once<T>&& other) = default;

	Once<T>& operator = (const Once<T>& other) = delete;

	Once<T>& operator = (Once<T>&& other) = default;

	operator T(void)
	{
		return obj_;
	}

	T get (void) const
	{
		return obj_;
	}

private:
	T obj_;
};

}

#endif // TEQ_ONCE_HPP
