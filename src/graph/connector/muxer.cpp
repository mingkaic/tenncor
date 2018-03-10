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

static inline std::vector<inode*> to_args (std::vector<MUXPAIR>& muxargs)
{
	std::vector<inode*> args(muxargs.size());
	std::transform(muxargs.begin(), muxargs.end(), args.begin(),
	[](MUXPAIR mpair) -> inode*
	{
		return mpair.first;
	});
	return args;
}

demuxer* demuxer::get (inode* arg, 
	NSLICE_F nslice, IDXS2ARR_F sliceidxer, 
	USHAPE_F sliceshaper, std::string label)
{ return new demuxer(arg, nslice, sliceidxer, sliceshaper, label); }

demuxer* demuxer::clone (void) const
{
	return static_cast<demuxer*>(clone_impl());
}

demuxer* demuxer::move (void)
{
	return static_cast<demuxer*>(move_impl());
}

demuxer& demuxer::operator = (const demuxer& other)
{
	if (this != &other)
	{
		iconnector::operator = (other);
		copy_helper(other);
	}
	return *this;
}

demuxer& demuxer::operator = (demuxer&& other)
{
	if (this != &other)
	{
		iconnector::operator = (std::move(other));
		move_helper(std::move(other));
	}
	return *this;
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
		size_t nslices = nslice_(shape);
		IDXS2ARR_F idxer = sliceidxer_;
		std::string label;
		for (size_t i = 0; i < nslices; ++i)
		{
			label = nnutils::formatter() << this->get_label() << "_slice" << i;
			inode* slice = single_parent(arg, label);
			if (nullptr == slice)
			{
				slice = coord_mapper::get(arg, [idxer, i](tensorshape outshape, const tensorshape inshape)
				{
					return idxer(outshape, inshape, i);
				}, sliceshaper_, label, true);
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


demuxer::demuxer (inode* arg, NSLICE_F nslice, IDXS2ARR_F sliceidxer, 
	USHAPE_F sliceshaper, std::string label) :
iconnector(std::vector<inode*>{arg}, label),
nslice_(nslice), sliceidxer_(sliceidxer), sliceshaper_(sliceshaper)
{ this->update(); }

demuxer::demuxer (const demuxer& other) : 
	iconnector(other)
{
	copy_helper(other);
}

demuxer::demuxer (demuxer&& other) : 
	iconnector(std::move(other))
{
	move_helper(std::move(other));
}


inode* demuxer::clone_impl (void) const
{
	return new demuxer(*this);
}

inode* demuxer::move_impl (void)
{
	return new demuxer(std::move(*this));
}


void demuxer::copy_helper (const demuxer& other)
{
	nslice_ = other.nslice_;
	sliceidxer_ = other.sliceidxer_;
	sliceshaper_ = other.sliceshaper_;
}

void demuxer::move_helper (demuxer&& other)
{
	nslice_ = std::move(other.nslice_);
	sliceidxer_ = std::move(other.sliceidxer_);
	sliceshaper_ = std::move(other.sliceshaper_);
}



muxer* muxer::get (std::vector<MUXPAIR> args, SHAPER_F shaper, MUXOP_F op, GLUE_F gluer, std::string label)
{
	return new muxer(args, shaper, op, gluer, label);
}

muxer* muxer::clone (void) const
{
	return static_cast<muxer*>(clone_impl());
}

muxer* muxer::move (void)
{
	return static_cast<muxer*>(move_impl());
}

muxer& muxer::operator = (const muxer& other)
{
	if (this != &other)
	{
		iconnector::operator = (other);
		copy_helper(other);
	}
	return *this;
}

muxer& muxer::operator = (muxer&& other)
{
	if (this != &other)
	{
		iconnector::operator = (std::move(other));
		move_helper(std::move(other));
	}
	return *this;
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
	GLUE_F gluer;
	auto wleaves = wrt->get_leaves();
	std::vector<inode*> temp;
	for (auto it = ggluer_.begin(), et = ggluer_.end();
		it != et && !gluer; it++)
	{
		auto mleaves = (*it).first->get_leaves();
		std::set_intersection(mleaves.begin(), mleaves.end(), 
			wleaves.begin(), wleaves.end(), std::back_inserter(temp));
		if (temp.size() > 0)
		{
			gluer = (*it).second;
		}
		temp.clear();
	}
	if (this == wrt || !gluer)
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

	std::vector<inode*> args = this->get_arguments();
	std::string glabel = "grad_" + this->get_label();
	if (inode* parent = ordered_parent(args, glabel))
	{
		return parent;
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
	return new muxer(ggluer_, gslices, 
	[shape](std::vector<tensorshape>)
	{
		return shape;
	}, op_, gluer, glabel);
}

void muxer::update (void)
{
	std::vector<inode*> args = this->get_arguments();
	bool has_data = std::all_of(args.begin(), args.end(), 
	[](inode* node)
	{
		tensor* tens = node->get_tensor();
		return nullptr != tens && tens->has_data();
	});
	if (has_data)
	{
		// build slices
		if (0 == slices_.size())
		{
			std::vector<std::vector<inode*> > vargs(args.size());
			std::transform(args.begin(), args.end(), vargs.begin(),
			[](inode* arg) -> std::vector<inode*> { return static_cast<demuxer*>(arg)->get_slices(); });
			slices_ = op_(vargs);
		}
		// piece together slices
		if (nullptr == data_)
		{
			std::vector<tensorshape> shapes(args.size());
			std::transform(args.begin(), args.end(), shapes.begin(),
			[](inode* node)
			{
				return node->get_tensor()->get_shape();
			});
			for (size_t i = 0; i < slices_.size(); ++i)
			{
				slices_[i]->get_tensor()->write_to(*gio_, i);
			}
			data_ = std::make_unique<tensor>(shaper_(shapes));
		}
		data_->read_from(*gio_);
		this->notify(UPDATE);
	}
}

muxer::muxer (std::vector<MUXPAIR> args, SHAPER_F shaper, 
	MUXOP_F op, GLUE_F gluer, std::string label) :
muxer(args, std::vector<varptr>{}, shaper, op, gluer, label) {}

muxer::muxer (std::vector<MUXPAIR> args, 
	std::vector<varptr> slices, SHAPER_F shaper, 
	MUXOP_F op, GLUE_F gluer, std::string label) : 
iconnector(to_args(args), label), 
gio_(new glue_io(gluer)), slices_(slices), 
ggluer_(args), shaper_(shaper), op_(op)
{ update(); }

muxer::muxer (const muxer& other) : 
	iconnector(other)
{
	copy_helper(other);
}

muxer::muxer (muxer&& other) : 
	iconnector(std::move(other))
{
	move_helper(std::move(other));
}

inode* muxer::clone_impl (void) const
{
	return new muxer(*this);
}

inode* muxer::move_impl (void)
{
	return new muxer(std::move(*this));
}


void muxer::copy_helper (const muxer& other)
{
	if (nullptr == other.data_)
	{
		data_ = nullptr;
	}
	else
	{
		data_ = std::make_unique<tensor>(*other.data_);
	}

	if (nullptr == other.gio_)
	{
		gio_ = nullptr;
	}
	else
	{
		gio_ = std::shared_ptr<glue_io>(other.gio_->clone());
	}

	slices_ = other.slices_;
	ggluer_ = other.ggluer_;
	shaper_ = other.shaper_;
	op_ = other.op_;
}

void muxer::move_helper (muxer&& other)
{
	data_ = std::move(other.data_);
	gio_ = std::move(other.gio_);
	slices_ = std::move(other.slices_);
	ggluer_ = std::move(other.ggluer_);
	shaper_ = std::move(other.shaper_);
	op_ = std::move(other.op_);
}

}

#endif
