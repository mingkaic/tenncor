
#ifndef TENNCOR_LAYER_HPP
#define TENNCOR_LAYER_HPP

#include "tenncor/layr/layer.hpp"

namespace tcr
{

template <typename T>
eteq::ETensor<T> connect (
	const eteq::ETensor<T>& root, const eteq::ETensor<T>& input)
{
	return layr::connect(root, input);
}

}

#endif // TENNCOR_LAYER_HPP
