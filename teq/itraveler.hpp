
#ifndef TEQ_ITRAVELER_HPP
#define TEQ_ITRAVELER_HPP

namespace teq
{

struct iLeaf;

struct iFunctor;

/// Interface to travel through graph, treating iLeaf and iFunctor differently
struct iTraveler
{
	virtual ~iTraveler (void) = default;

	/// Visit leaf node
	virtual void visit (iLeaf& leaf) = 0;

	/// Visit functor node
	virtual void visit (iFunctor& func) = 0;
};

}

#endif // TEQ_ITRAVELER_HPP
