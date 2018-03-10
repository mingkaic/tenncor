#include "mocker/mocker.hpp"

#include "graph/react/iobserver.hpp"

#ifndef TTEST_MOCK_OBSERVER_HPP
#define TTEST_MOCK_OBSERVER_HPP

namespace testutils
{

class mock_observer final : public nnet::iobserver, public testify::mocker
{
public:
	// trust tester to allocate on stack
	mock_observer (void) : nnet::iobserver() {}

	mock_observer (std::vector<nnet::subject*> args) : nnet::iobserver(args)  {}

	virtual void update (void)
	{
		label_incr("update1");
	}

	virtual void update (nnet::notification msg, std::unordered_set<size_t> indices)
	{
		nnet::iobserver::update(msg, indices);
		label_incr("update2");
		switch (msg)
		{
			case nnet::UNSUBSCRIBE:
				set_label("update2", "UNSUBSCRIBE");
				break;
			case nnet::UPDATE:
				set_label("update2", "UPDATE");
				break;
		}
	}
	
	virtual void death_on_broken (void)
	{
		label_incr("death_on_broken");
	}
	
	void mock_add_dependency (nnet::subject* dep)
	{
		this->add_dependency(dep);
	}
	
	void mock_remove_dependency (size_t idx)
	{
		this->remove_dependency(idx);
	}
	
	void mock_replace_dependency (nnet::subject* dep, size_t i)
	{
		this->replace_dependency(dep, i);
	}

	void mock_clear_dependency (void)
	{
		this->dependencies_.clear();
	}

	std::vector<nnet::subject*> expose_dependencies (void)
	{
		return this->dependencies_;
	}
};

}

#endif
