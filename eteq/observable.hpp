#ifndef ETEQ_OBSERVABLE_HPP
#define ETEQ_OBSERVABLE_HPP

#include <unordered_set>

#include "teq/ifunctor.hpp"

namespace eteq
{

struct Observable : public teq::iFunctor
{
	virtual ~Observable (void) = default;

	void subscribe (Observable* sub)
	{
		subs_.emplace(sub);
	}

	void unsubscribe (Observable* sub)
	{
		subs_.erase(sub);
	}

	virtual bool has_data (void) const = 0;

	/// Removes internal data object
	virtual void uninitialize (void) = 0;

	/// Best effort internal data object initialization
	/// Return true if initialized, otherwise false
	/// Will not recursively initialize children
	virtual bool initialize (void) = 0;

	/// Do or die populate internal data object, will recurse
	virtual void must_initialize (void) = 0;

protected:
	std::unordered_set<Observable*> subs_;
};

using ObsptrT = std::shared_ptr<Observable>;

}

#endif // ETEQ_OBSERVABLE_HPP
