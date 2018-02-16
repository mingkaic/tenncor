/*!
 *
 *  coord_mapper.hpp
 *  cnnet
 *
 *  Purpose:
 *  maps source data 1 to 1 with dest data via index map
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/connector/immutable/immutable.hpp"

#pragma once
#ifndef TENNCOR_COORD_MAPPER_HPP
#define TENNCOR_COORD_MAPPER_HPP

namespace nnet
{

class coord_mapper final : public immutable
{
public:
	// >>>>>>>>>>>> BUILDER TO FORCE HEAP ALLOCATION <<<<<<<<<<<<

	// input smap must transform shape to desired output shape
	// smap returns vector of size dest, containing index value pointing to input
	static coord_mapper* get (inode* arg, SIDX_F smap, USHAPE_F shaper, std::string name)
	{
		return new coord_mapper(arg, smap, shaper, name);
	}

	//! clone function
	coord_mapper* clone (void) const
	{
		return static_cast<coord_mapper*>(clone_impl());
	}

	//! move function
	coord_mapper* move (void)
	{
		return static_cast<coord_mapper*>(move_impl());
	}

	//! declare copy assignment to copy over transfer functions
	virtual coord_mapper& operator = (const coord_mapper& other)
	{
		if (this != &other)
		{
			immutable::operator = (other);
			copy_helper(other);
		}
		return *this;
	}

	//! declare move assignment to move over transfer functions
	virtual coord_mapper& operator = (coord_mapper&& other)
	{
		if (this != &other)
		{
			immutable::operator = (std::move(other));
			move_helper(std::move(other));
		}
		return *this;
	}

private:
	coord_mapper (inode* arg, SIDX_F smap, USHAPE_F shaper, std::string label) :
		coord_mapper(arg, std::make_shared<sindex_io>(smap), shaper, label) {}

	coord_mapper (inode* arg, std::shared_ptr<sindex_io> sio, USHAPE_F shaper, std::string label) :
		immutable(std::vector<inode*>{arg}, label), sio_(sio), shaper_(shaper)
	{ this->update(); }

	coord_mapper (const coord_mapper& other) : immutable(other)
	{
		copy_helper(other);
	}

	coord_mapper (coord_mapper&& other) : immutable(std::move(other))
	{
		move_helper(std::move(other));
	}

	// >>>>>> POLYMORPHIC CLONERS <<<<<<

	//! implement clone function
	virtual inode* clone_impl (void) const
	{
		return new coord_mapper(*this);
	}

	//! move implementation
	virtual inode* move_impl (void)
	{
		return new coord_mapper(std::move(*this));
	}

	// >>>>>> FORWARD & BACKWARD <<<<<<

	//! forward pass step: populate data_
	virtual void forward_pass (std::vector<inode*>& args)
	{
		if (nullptr == data_)
		{
			tensor* ten = args[0]->get_tensor();
			if (nullptr == ten)
			{
				throw std::exception(); // todo: better exception
			}
			// assert that shape only change once
			ten->write_to(*sio_);
			data_ = std::make_unique<tensor>(shaper_(ten->get_shape()));
		}
		data_->read_from(*sio_);
	}

	//! backward pass step
	virtual varptr backward_pass (inode* wrt)
	{
		return new coord_mapper(this->get_arguments()[0]->derive(wrt), sio_, shaper_, this->get_label());
	}


	//! copy helper
	void copy_helper (const coord_mapper& other)
	{
		if (nullptr == other.sio_)
		{
			sio_ = nullptr;
		}
		else
		{
			sio_ = std::shared_ptr<sindex_io>(other.sio_->clone());
		}
		shaper_ = other.shaper_;
	}

	//! move helper
	void move_helper (coord_mapper&& other)
	{
		sio_ = std::move(other.sio_);
		shaper_ = std::move(other.shaper_);
	}

	std::shared_ptr<sindex_io> sio_;

	USHAPE_F shaper_;
};

}

#endif /* TENNCOR_COORD_MAPPER_HPP */
