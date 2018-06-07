//
//  ondeath.cpp
//  mold
//

#include "mold/ondeath.hpp"

#ifdef MOLD_ONDEATH_HPP

namespace mold
{

OnDeath::OnDeath (iNode* arg, TermF term) :
	iObserver({arg}), terminate_(term) {}

OnDeath::~OnDeath (void)
{
	terminate_();
}

OnDeath::OnDeath (const OnDeath& other, TermF term) :
	iObserver(other), terminate_(term) {}

OnDeath::OnDeath (OnDeath&& other, TermF term) :
	iObserver(std::move(other)), terminate_(std::move(term)) {}

iNode* OnDeath::get (void) const
{
	iNode* out = nullptr;
	if (false == this->args_.empty())
	{
		out = this->args_[0];
	}
	return out;
}

void OnDeath::initialize (void) {}

void OnDeath::update (void) {}

void OnDeath::clear_term (void)
{
	terminate_ = [](){};
}

}

#endif
