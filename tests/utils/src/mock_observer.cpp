#include "mock_observer.hpp"

#ifdef TTEST_MOCK_OBSERVER_HPP

namespace testutils
{

mock_observer::mock_observer (void) : nnet::iobserver() {}

mock_observer::mock_observer (std::vector<nnet::subject*> args) : nnet::iobserver(args)  {}

void mock_observer::update (void)
{
	label_incr("update1");
}

void mock_observer::update (nnet::notification msg, std::unordered_set<size_t> indices)
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

void mock_observer::death_on_broken (void)
{
	label_incr("death_on_broken");
}

void mock_observer::mock_add_dependency (nnet::subject* dep)
{
	this->add_dependency(dep);
}

void mock_observer::mock_remove_dependency (size_t idx)
{
	this->remove_dependency(idx);
}

void mock_observer::mock_replace_dependency (nnet::subject* dep, size_t i)
{
	this->replace_dependency(dep, i);
}

void mock_observer::mock_clear_dependency (void)
{
	this->dependencies_.clear();
}

std::vector<nnet::subject*> mock_observer::expose_dependencies (void)
{
	return this->dependencies_;
}

}

#endif
