#include "llo/node.hpp"

#ifndef LLO_API_HPP
#define LLO_API_HPP

namespace llo
{

ade::Tensorptr abs (ade::Tensorptr& arg);

ade::Tensorptr neg (ade::Tensorptr& arg);

ade::Tensorptr bit_not (ade::Tensorptr& arg);

ade::Tensorptr sin (ade::Tensorptr& arg);

ade::Tensorptr cos (ade::Tensorptr& arg);

ade::Tensorptr tan (ade::Tensorptr& arg);

ade::Tensorptr exp (ade::Tensorptr& arg);

ade::Tensorptr log (ade::Tensorptr& arg);

ade::Tensorptr sqrt (ade::Tensorptr& arg);

ade::Tensorptr round (ade::Tensorptr& arg);

ade::Tensorptr flip (ade::Tensorptr& arg, uint8_t dim);

ade::Tensorptr pow (ade::Tensorptr& a, ade::Tensorptr& b);

ade::Tensorptr add (ade::Tensorptr& a, ade::Tensorptr& b);

ade::Tensorptr sub (ade::Tensorptr& a, ade::Tensorptr& b);

ade::Tensorptr mul (ade::Tensorptr& a, ade::Tensorptr& b);

ade::Tensorptr div (ade::Tensorptr& a, ade::Tensorptr& b);

ade::Tensorptr eq (ade::Tensorptr& a, ade::Tensorptr& b);

ade::Tensorptr neq (ade::Tensorptr& a, ade::Tensorptr& b);

ade::Tensorptr lt (ade::Tensorptr& a, ade::Tensorptr& b);

ade::Tensorptr gt (ade::Tensorptr& a, ade::Tensorptr& b);

ade::Tensorptr binom (ade::Tensorptr& ntrials, ade::Tensorptr& prob);

ade::Tensorptr uniform (ade::Tensorptr& lower, ade::Tensorptr& upper);

ade::Tensorptr normal (ade::Tensorptr& mean, ade::Tensorptr& stdev);

ade::Tensorptr n_elems (ade::Tensorptr& arg);

ade::Tensorptr n_dims (ade::Tensorptr& arg, uint8_t dim);

ade::Tensorptr argmax (ade::Tensorptr& arg);

ade::Tensorptr rmax (ade::Tensorptr& arg);

ade::Tensorptr rsum (ade::Tensorptr& arg);

ade::Tensorptr matmul (ade::Tensorptr& a, ade::Tensorptr& b);

ade::Tensorptr matmul (ade::Tensorptr& a, ade::Tensorptr& b,
	uint8_t agroup_idx, uint8_t bgroup_idx);

ade::Tensorptr convolute (ade::Tensorptr& canvas, ade::Tensorptr& window);

ade::Tensorptr permute (ade::Tensorptr& arg, std::vector<uint8_t> order);

ade::Tensorptr extend (ade::Tensorptr& arg, std::vector<uint8_t> ext);

ade::Tensorptr reshape (ade::Tensorptr& arg, std::vector<uint8_t> slist);

}

#endif /* LLO_API_HPP */
