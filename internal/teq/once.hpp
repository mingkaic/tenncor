///
/// once.hpp
/// teq
///
/// Purpose:
/// Define holder object that calls signal once upon destruction
///

#ifndef TEQ_ONCE_HPP
#define TEQ_ONCE_HPP

#include <functional>

// from cppkg/jobs (upgrade when ready)
namespace jobs
{

using GuardOpF = std::function<void(void)>;

struct ScopeGuard2
{
	ScopeGuard2 (GuardOpF f) : term_(f) {}

	virtual ~ScopeGuard2 (void)
	{
		if (term_)
		{
			term_();
		}
	}

	ScopeGuard2 (const ScopeGuard2&) = delete;

	ScopeGuard2 (ScopeGuard2&& other) :
		term_(std::move(other.term_)) {}

	ScopeGuard2& operator = (const ScopeGuard2&) = delete;

	ScopeGuard2& operator = (ScopeGuard2&& other)
	{
		if (this != &other)
		{
			if (term_)
			{
				term_();
			}
			term_ = std::move(other.term_);
		}
		return *this;
	}

private:
	GuardOpF term_;
};

}

namespace teq
{

template <typename T>
struct Once final : public jobs::ScopeGuard2
{
	Once (T obj, jobs::GuardOpF killsig = jobs::GuardOpF()) :
		ScopeGuard2(killsig), obj_(obj) {}

	template <typename OT>
	Once (T altobj, Once<OT>&& other) :
		ScopeGuard2(std::move(other)), obj_(altobj) {}

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
