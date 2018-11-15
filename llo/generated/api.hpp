#include "age/runtime/grader.hpp"
#include "ade/functor.hpp"

#ifndef _GENERATED_API_HPP
#define _GENERATED_API_HPP

namespace age
{

ade::Tensorptr abs (ade::Tensorptr arg1);

ade::Tensorptr neg (ade::Tensorptr arg1);

ade::Tensorptr sin (ade::Tensorptr arg1);

ade::Tensorptr cos (ade::Tensorptr arg1);

ade::Tensorptr tan (ade::Tensorptr arg1);

ade::Tensorptr exp (ade::Tensorptr arg1);

ade::Tensorptr log (ade::Tensorptr arg1);

ade::Tensorptr sqrt (ade::Tensorptr arg1);

ade::Tensorptr round (ade::Tensorptr arg1);

ade::Tensorptr flip (ade::Tensorptr arg1, uint8_t arg2);

ade::Tensorptr pow (ade::Tensorptr arg1, ade::Tensorptr arg2);

ade::Tensorptr add (ade::Tensorptr arg1, ade::Tensorptr arg2);

ade::Tensorptr sub (ade::Tensorptr arg1, ade::Tensorptr arg2);

ade::Tensorptr mul (ade::Tensorptr arg1, ade::Tensorptr arg2);

ade::Tensorptr div (ade::Tensorptr arg1, ade::Tensorptr arg2);

ade::Tensorptr eq (ade::Tensorptr arg1, ade::Tensorptr arg2);

ade::Tensorptr neq (ade::Tensorptr arg1, ade::Tensorptr arg2);

ade::Tensorptr lt (ade::Tensorptr arg1, ade::Tensorptr arg2);

ade::Tensorptr gt (ade::Tensorptr arg1, ade::Tensorptr arg2);

ade::Tensorptr rand_bino (ade::Tensorptr arg1, ade::Tensorptr arg2);

ade::Tensorptr rand_unif (ade::Tensorptr arg1, ade::Tensorptr arg2);

ade::Tensorptr rand_norm (ade::Tensorptr arg1, ade::Tensorptr arg2);

ade::Tensorptr sum (age::TensT arg1);

ade::Tensorptr prod (age::TensT arg1);

ade::Tensorptr min (age::TensT arg1);

ade::Tensorptr max (age::TensT arg1);

ade::Tensorptr reduce_sum (ade::Tensorptr arg1, uint8_t arg2);

ade::Tensorptr reduce_min (ade::Tensorptr arg1, uint8_t arg2);

ade::Tensorptr reduce_max (ade::Tensorptr arg1, uint8_t arg2);

ade::Tensorptr permute (ade::Tensorptr arg1, std::vector<uint8_t> arg2);

ade::Tensorptr extend (ade::Tensorptr arg1, uint8_t arg2, std::vector<uint8_t> arg3);

ade::Tensorptr reduce_sum (ade::Tensorptr arg1);

ade::Tensorptr reduce_min (ade::Tensorptr arg1);

ade::Tensorptr reduce_max (ade::Tensorptr arg1);

ade::Tensorptr matmul (ade::Tensorptr arg1, ade::Tensorptr arg2);

ade::Tensorptr convolute (ade::Tensorptr arg1, ade::Tensorptr arg2);


}

#endif // _GENERATED_API_HPP
