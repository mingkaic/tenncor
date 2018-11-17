#include <cassert>
#include "age/test/grader_dep.hpp"

ade::Tensorptr arms_heavy (size_t idx, age::TensT args)
{
	assert(args.size() > 0);
	static_cast<MockTensor*>(args[0].get())->scalar_ = idx;
	return args[0];
}

ade::Tensorptr dj_grad (age::TensT args, size_t idx)
{
	assert(args.size() > 0);
	static_cast<MockTensor*>(args[0].get())->scalar_ = idx + khaled_constant;
	return args[0];
}
