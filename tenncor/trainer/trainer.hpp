
#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "tenncor/layr/layer.hpp"
#include "tenncor/layr/approx.hpp"

#include "tenncor/tenncor.hpp"

namespace trainer
{

template <typename T>
using TrainErrF = std::function<teq::ShapedArr<T>(void)>;

}

#endif // TRAINER_HPP
