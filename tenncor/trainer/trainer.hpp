
#include "layr/layer.hpp"
#include "layr/approx.hpp"

#include "tenncor/tenncor.hpp"

#ifndef TRAINER_HPP
#define TRAINER_HPP

namespace trainer
{

template <typename T>
using TrainErrF = std::function<teq::ShapedArr<T>(void)>;

}

#endif // TRAINER_HPP
