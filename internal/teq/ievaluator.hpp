
#ifndef TEQ_IEVALUATOR_HPP
#define TEQ_IEVALUATOR_HPP

#include "internal/teq/itensor.hpp"

namespace teq
{

struct iDevice
{
	virtual ~iDevice (void) = default;

	virtual void calc (iTensor& tens, size_t cache_ttl) = 0;
};

struct iEvaluator
{
	virtual ~iEvaluator (void) = default;

	virtual void evaluate (
		iDevice& device,
		const TensSetT& targets,
		const TensSetT& ignored = {}) = 0;
};

using iEvalptrT = std::shared_ptr<iEvaluator>;

}

#endif // TEQ_IEVALUATOR_HPP
