#include "mocker/mocker.hpp"

#include "graph/react/subject.hpp"

#ifndef TTEST_MOCK_SUBJECT_HPP
#define TTEST_MOCK_SUBJECT_HPP

namespace testutils
{

class mock_subject final : public nnet::subject, public testify::mocker
{
public:
	mock_subject (void) {}

	mock_subject (const mock_subject& other) : nnet::subject(other) {}

	mock_subject (mock_subject&& other) : nnet::subject(std::move(other)) {}

	mock_subject& operator = (const mock_subject& other)
	{
		nnet::subject::operator = (other);
		return *this;
	}

	mock_subject& operator = (mock_subject&& other)
	{
		nnet::subject::operator = (std::move(other));
		return *this;
	}

	void attach (nnet::iobserver* viewer, size_t idx)
	{
		nnet::subject::attach(viewer, idx);
	}

	virtual void detach (nnet::iobserver* viewer)
	{
		subject::detach(viewer);
		label_incr("detach1");
	}

	virtual void detach (nnet::iobserver* viewer, size_t idx)
	{
		subject::detach(viewer, idx);
		label_incr("detach2");
	}
};

}

#endif
