
#include "teq/itensor.hpp"

#ifndef TEQ_IEVALUATOR_HPP
#define TEQ_IEVALUATOR_HPP

namespace teq
{

struct iDevice
{
	virtual ~iDevice (void) = default;

	virtual void calc (iTensor& tens) = 0;
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
