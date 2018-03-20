#include "mock_subject.hpp"

#ifdef TTEST_MOCK_SUBJECT_HPP

namespace testutils
{

mock_subject::mock_subject (void) {}

mock_subject::mock_subject (const mock_subject& other) : nnet::subject(other) {}

mock_subject::mock_subject (mock_subject&& other) : nnet::subject(std::move(other)) {}

mock_subject& mock_subject::operator = (const mock_subject& other)
{
	nnet::subject::operator = (other);
	return *this;
}

mock_subject& mock_subject::operator = (mock_subject&& other)
{
	nnet::subject::operator = (std::move(other));
	return *this;
}

void mock_subject::attach (nnet::iobserver* viewer, size_t idx)
{
	nnet::subject::attach(viewer, idx);
}

void mock_subject::detach (nnet::iobserver* viewer)
{
	subject::detach(viewer);
	label_incr("detach1");
}

void mock_subject::detach (nnet::iobserver* viewer, size_t idx)
{
	subject::detach(viewer, idx);
	label_incr("detach2");
}

}

#endif
