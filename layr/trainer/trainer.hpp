#include "eteq/derive.hpp"

#include "layr/api.hpp"
#include "layr/approx.hpp"

#ifndef TRAINER_HPP
#define TRAINER_HPP

namespace trainer
{

template <typename T>
using TrainErrF = std::function<teq::ShapedArr<T>(void)>;

}

#endif // TRAINER_HPP
