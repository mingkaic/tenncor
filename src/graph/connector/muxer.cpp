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

demuxer* demuxer::get (inode* arg, 
	SLICESHAPER_F sliceshaper, 
	SLICER_F slicer, std::string label)
{ return new demuxer(arg, sliceshaper, slicer, label); }

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


std::unordered_set<ileaf*> demuxer::get_leaves (void) const
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
	this->notify(UPDATE);
}

std::vector<inode*> demuxer::get_slices (void)
{
	std::vector<inode*> out;
	inode* arg = this->get_arguments()[0];
	if (tensor* ten = arg->get_tensor())
	{
		tensorshape shape = ten->get_shape();
		size_t nslices = sliceshaper_(shape);
		for (size_t i = 0; i < nslices; ++i)
		{
			out.push_back(coord_mapper::get(arg, 
			[this, shape, i](tensorshape& outshape)
			{
				std::vector<size_t> indices = slicer_(outshape, i);
				outshape = shape;
				return indices;
			}, nnutils::formatter() << this->get_label() << "_slice" << i));
		}
		if (0 == nslices)
		{
			out.push_back(arg);
		}
	}
	return out;
}


demuxer::demuxer (inode* arg, SLICESHAPER_F sliceshaper, 
	SLICER_F slicer, std::string label) :
iconnector(std::vector<inode*>{arg}, label),
sliceshaper_(sliceshaper), slicer_(slicer)
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
	sliceshaper_ = other.sliceshaper_;
	slicer_ = other.slicer_;
}

void demuxer::move_helper (demuxer&& other)
{
	sliceshaper_ = std::move(other.sliceshaper_);
	slicer_ = std::move(other.slicer_);
}



muxer* muxer::get (std::vector<demuxer*> args, SHAPER_F shaper, VAROP_F op, GLUE_F gluer, std::string label)
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


std::unordered_set<ileaf*> muxer::get_leaves (void) const
{
	return this->get_arguments()[0]->get_leaves();
}

tensor* muxer::get_tensor (void)
{
	return data_.get();
}

varptr muxer::derive (inode* wrt)
{
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
	std::vector<demuxer*> muxargs(args.size());
	std::transform(args.begin(), args.end(), muxargs.begin(),
	[](inode* node)
	{
		return static_cast<demuxer*>(node);
	});
	return new muxer(muxargs, gslices, shaper_, op_, gio_, glabel);
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
			std::vector<std::vector<varptr> > vargs;
			for (inode* arg : args)
			{
				demuxer* dmx = static_cast<demuxer*>(arg);
				std::vector<inode*> slices = dmx->get_slices();
				if (vargs.empty())
				{
					vargs = std::vector<std::vector<varptr> >(slices.size());
				}
				for (size_t i = 0; i < slices.size(); ++i)
				{
					vargs[i].push_back(slices[i]);
				}
			}
			for (std::vector<varptr>& varg : vargs)
			{
				slices_.push_back(op_(varg));
			}
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


muxer::muxer (std::vector<demuxer*> args, SHAPER_F shaper, 
	VAROP_F op, GLUE_F gluer, std::string label) : 
muxer(args, std::vector<varptr>{}, shaper, op, 
	std::make_shared<glue_io>(gluer), label) {}

muxer::muxer (std::vector<demuxer*> args, std::vector<varptr> slices, 
	SHAPER_F shaper, VAROP_F op, std::shared_ptr<glue_io> gio, std::string label) :
iconnector(std::vector<inode*>(args.begin(), args.end()), label), 
gio_(gio), slices_(slices), shaper_(shaper), op_(op) { update(); }

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
	op_ = other.op_;
}

void muxer::move_helper (muxer&& other)
{
	data_ = std::move(other.data_);
	gio_ = std::move(other.gio_);
	slices_ = std::move(other.slices_);
	op_ = std::move(other.op_);
}

}

#endif
