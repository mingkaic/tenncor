#include "testify_cpp/include/mocker/mocker.hpp"

#include "graph/react/subject.hpp"

#ifndef TTEST_MOCK_SUBJECT_HPP
#define TTEST_MOCK_SUBJECT_HPP

namespace testutils
{

class mock_subject final : public nnet::subject, public testify::mocker
{
public:
	mock_subject (void);

	mock_subject (const mock_subject& other);

	mock_subject (mock_subject&& other);

	mock_subject& operator = (const mock_subject& other);

	mock_subject& operator = (mock_subject&& other);

	void attach (nnet::iobserver* viewer, size_t idx);

	virtual void detach (nnet::iobserver* viewer);

	virtual void detach (nnet::iobserver* viewer, size_t idx);
};

}

#endif /* TTEST_MOCK_SUBJECT_HPP */
