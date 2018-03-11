//
// Created by Mingkai Chen on 2017-07-03.
//

#include "include/graph/connector/shape_dep.hpp"

#ifdef TENNCOR_SHAPE_DEP_HPP

namespace nnet
{

functor* shape_func (inode* arg, USIDX_F extracter, USHAPE_F shaper, std::string label)
{
	return functor::get({arg},
	[extracter, shaper](std::unique_ptr<idata_src>& src, std::vector<inode*> args) -> tensor*
	{
		tensor* tens = args[0]->get_tensor();
		assert(nullptr != tens);
		const_init* ci = new const_init();
		src = std::unique_ptr<idata_src>(ci);

		tensorshape shape = tens->get_shape();
		std::vector<size_t> sdata = extracter(shape);
		std::vector<double> doub_d(sdata.begin(), sdata.end()); // todo: make tens's type
		ci->set<double>(doub_d);
		return new tensor(shaper(shape));
	},
	[](inode*, std::vector<inode*>) -> varptr
	{
		return nullptr;
	}, label);
}

}

#endif
