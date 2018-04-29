#include "testify_cpp/include/mocker/mocker.hpp"

#include "graph/react/iobserver.hpp"

#ifndef TTEST_MOCK_OBSERVER_HPP
#define TTEST_MOCK_OBSERVER_HPP

namespace testutils
{

class mock_observer final : public nnet::iobserver, public testify::mocker
{
public:
	// trust tester to allocate on stack
	mock_observer (void);

	mock_observer (std::vector<nnet::subject*> args);

	virtual void update (void);

	virtual void update (nnet::notification msg, std::unordered_set<size_t> indices);
	
	virtual void death_on_broken (void);
	
	void mock_add_dependency (nnet::subject* dep);
	
	void mock_remove_dependency (size_t idx);
	
	void mock_replace_dependency (nnet::subject* dep, size_t i);

	void mock_clear_dependency (void);

	std::vector<nnet::subject*> expose_dependencies (void);
};

}

#endif
