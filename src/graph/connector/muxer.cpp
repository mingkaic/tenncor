//
//  muxer.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-12.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/connector/muxer.hpp"
#include "include/operations/operation_utils.hpp"

#ifdef TENNCOR_MUXER_HPP

namespace nnet
{

demuxer* demuxer::get (inode* arg, size_t dimension, std::string label)
{ return new demuxer(arg, dimension, label); }

demuxer* demuxer::clone (void) const
{
	return static_cast<demuxer*>(clone_impl());
}

demuxer* demuxer::move (void)
{
	return static_cast<demuxer*>(move_impl());
}


std::unordered_set<inode*> demuxer::get_leaves (void) const
{
	return this->get_arguments()[0]->get_leaves();
}

tensor* demuxer::get_tensor (void)
{
	return this->get_arguments()[0]->get_tensor();
}

varptr demuxer::derive (inode* wrt)
{
	return this->get_arguments()[0]->derive(wrt);
}

void demuxer::update (void)
{
	this->notify(UPDATE); // todo: wait for slices to update
}

std::vector<inode*> demuxer::get_slices (void)
{
	std::vector<inode*> out;
	inode* arg = this->get_arguments()[0];
	if (tensor* ten = arg->get_tensor())
	{
		tensorshape shape = ten->get_shape();
		size_t rank = shape.rank();
		size_t nslices = 0;
		std::vector<size_t> slist = shape.as_list();
		if (rank > dimension_)
		{
			slist[dimension_] = 1;
			nslices = std::accumulate(slist.begin(), slist.end(), 
				(size_t) 1, std::multiplies<size_t>());
		}
		std::string label;
		for (size_t idx = 0; idx < nslices; ++idx)
		{
			label = nnutils::formatter() << this->get_label() << "_slice" << idx;
			inode* slice = single_parent(arg, label);
			if (nullptr == slice)
			{
				slice = coord_func(arg, [idx, this](tensorshape outshape, const tensorshape inshape)
				{
					size_t n = outshape.n_elems();
					size_t rank = inshape.rank();
					assert(rank > dimension_);
					std::vector<size_t> slist = inshape.as_list();
					slist[dimension_] = 1;
					std::vector<size_t> coords = tensorshape(slist).coord_from_idx(idx);
					std::vector<size_t> out(n);
					for (size_t j = 0; j < n; ++j)
					{
						coords[dimension_] = j;
						out[j] = inshape.flat_idx(coords);
					}
					return out;
				}, [this](tensorshape inshape)
				{
					assert(inshape.rank() > dimension_);
					size_t slimit = inshape.as_list()[dimension_];
					return tensorshape(std::vector<size_t>{slimit});
				}, label, true);
			}
			out.push_back(slice);
		}
		if (0 == nslices)
		{
			out.push_back(arg);
		}
	}
	return out;
}


demuxer::demuxer (inode* arg, size_t dimension, std::string label) :
iconnector(std::vector<inode*>{arg}, label), dimension_(dimension)
{ this->update(); }


inode* demuxer::clone_impl (void) const
{
	return nullptr;
}

inode* demuxer::move_impl (void)
{
	return nullptr;
}



muxer* muxer::get (inode* arg, size_t dimension, std::function<varptr(varptr)> reduce_op, std::string label)
{
	return new muxer(arg, dimension, reduce_op, label);
}

muxer* muxer::clone (void) const
{
	return static_cast<muxer*>(clone_impl());
}

muxer* muxer::move (void)
{
	return static_cast<muxer*>(move_impl());
}


std::unordered_set<inode*> muxer::get_leaves (void) const
{
	return this->get_arguments()[0]->get_leaves();
}

tensor* muxer::get_tensor (void)
{
	return data_.get();
}

varptr muxer::derive (inode* wrt)
{
	// find a suitable gluer
	if (this == wrt)
	{
		tensor* data = wrt->get_tensor();
		if (data && data->has_data())
		{
			tensorshape shape = data->get_shape();
			std::vector<double> zeroes(shape.n_elems(), (double) (this == wrt)); // todo: convert to data type
			return constant::get(zeroes, shape);
		}
		return nullptr;
	}

	std::vector<varptr> gslices(slices_.size());
	std::transform(slices_.begin(), slices_.end(), gslices.begin(),
	[wrt](varptr slice)
	{
		return slice->derive(wrt);
	});
	// clone a version of this using gslices
	tensor* ten = wrt->get_tensor();
	assert(ten && ten->has_data());
	tensorshape shape = ten->get_shape();
	std::string glabel = "grad_" + get_label();
	return new muxer(static_cast<demuxer*>(get_arguments()[0]), dimension_, reduce_op_, shape, gslices, glabel);
}

void muxer::update (void)
{
	inode* arg = this->get_arguments()[0];
	inode* orig = this->get_arguments()[1];
	if (arg->get_tensor()->has_data())
	{
		// build slices
		if (0 == slices_.size())
		{
			// assert sliceargs.size() == 1
			std::vector<inode*> sarg = static_cast<demuxer*>(arg)->get_slices();
			for (inode* varg : sarg)
			{
				slices_.push_back(reduce_op_(varg));
			}
		}
		// piece together slices
		if (nullptr == data_)
		{
			for (size_t i = 0; i < slices_.size(); ++i)
			{
				slices_[i]->get_tensor()->write_to(*gio_, i);
			}
	
			if (shape_.is_fully_defined())
			{
				data_ = std::make_unique<tensor>(shape_);
			}
			else
			{
				std::vector<size_t> slist = orig->get_tensor()->get_shape().as_list();
				if (1 == slist.size())
				{
					slist[0] = 1;
				}
				else
				{
					slist.erase(slist.begin() + dimension_);
				}
				data_ = std::make_unique<tensor>(tensorshape(slist));
			}
		}
		data_->read_from(*gio_);
		this->notify(UPDATE);
	}
}

muxer::muxer (inode* arg, size_t dimension, std::function<varptr(varptr)> reduce_op, std::string label) :
	muxer(arg, dimension, reduce_op, tensorshape(), std::vector<varptr>{}, label) {}

muxer::muxer (inode* arg, size_t dimension, std::function<varptr(varptr)> reduce_op,
	tensorshape shape,
	std::vector<varptr> slices, std::string label) : 
iconnector({demuxer::get(arg, dimension, label), arg}, label), dimension_(dimension), shape_(shape),
reduce_op_(reduce_op),
gio_(new glue_io([dimension](VARR_T dest, CVAR_T src, unsigned short bytesize, size_t i)
{
	char* cdest = (char*) dest.first;
	const char* csrc = (const char*) src.first;
	size_t srcn = src.second.n_elems();
	if (dest.second.rank() > 1)
	{
		std::vector<size_t> slist = dest.second.as_list();
		slist[dimension] = 1;
		std::vector<size_t> coords = tensorshape(slist).coord_from_idx(i);
		size_t desti = 0;
		for (size_t srci = 0; srci < srcn; ++srci)
		{
			coords[dimension] = srci;
			desti = dest.second.flat_idx(coords);
			memcpy(cdest + desti * bytesize, csrc + srci * bytesize, bytesize);
		}
	}
	else
	{
		memcpy(cdest + i * srcn * bytesize, csrc, srcn * bytesize);
	}
})), slices_(slices)
{ update(); }

inode* muxer::clone_impl (void) const
{
	return nullptr;
}

inode* muxer::move_impl (void)
{
	return nullptr;
}

}

#endif
