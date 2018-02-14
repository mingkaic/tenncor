/*!
 *
 *  muxer.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph muxer isolates a tensor's dimensions
 *  the caller can treat each dimension as a separate connector node
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/connector/immutable/coord_mapper.hpp"

#pragma once
#ifndef TENNCOR_MUXER_HPP
#define TENNCOR_MUXER_HPP

namespace nnet
{

using SLICESHAPER_F = std::function<size_t(tensorshape&)>;

using SLICER_F = std::function<std::vector<size_t>(tensorshape,size_t)>;

using VAROP_F = std::function<varptr(std::vector<varptr>)>;

class demuxer final : public iconnector
{
public:
	static demuxer* get (inode* arg, 
		SLICESHAPER_F slicershaper, 
		SLICER_F slicer, std::string label);
	
	//! clone function
	demuxer* clone (void) const;
	
	//! move function
	demuxer* move (void);

	//! declare copy assignment to copy over transfer functions
	virtual demuxer& operator = (const demuxer& other);

	//! declare move assignment to move over transfer functions
	virtual demuxer& operator = (demuxer&& other);



	// >>>>>>>>>>>> ACCESSORS <<<<<<<<<<<<
	
	//! get gradient leaves
	virtual std::unordered_set<ileaf*> get_leaves (void) const;



	// >>>>>>>>>>>> MUTATORS <<<<<<<<<<<<

	//! get tensor data
	virtual tensor* get_tensor (void);

	//! get gradient wrt some node, applies jacobians before evaluting resulting tensor
	//! may call get_gradient
	virtual varptr derive (inode* wrt);

	// >>>>>> CALLED BY OBSERVER TO UPDATE <<<<<<

	//! Inherited from iobserver: update data
	virtual void update (void);

	std::vector<inode*> get_slices (void);

private:
	demuxer (inode* arg, SLICESHAPER_F sliceshaper, 
		SLICER_F slicer, std::string label);

	demuxer (const demuxer& other);

	demuxer (demuxer&& other);

	// >>>>>> POLYMORPHIC CLONERS <<<<<<

	//! implement clone function
	virtual inode* clone_impl (void) const;

	//! move implementation
	virtual inode* move_impl (void);


	//! copy helper
	void copy_helper (const demuxer& other);

	//! move helper
	void move_helper (demuxer&& other);

	SLICESHAPER_F sliceshaper_;

	SLICER_F slicer_;
};

// consumes the volatile consumers of demuxer's slices without suiciding
class muxer final : public iconnector
{
public:
	static muxer* get (std::vector<demuxer*> args, 
		SHAPER_F shaper, VAROP_F op, GLUE_F gluer, std::string label);
	
	//! clone function
	muxer* clone (void) const;

	//! move function
	muxer* move (void);

	//! declare copy assignment to copy over transfer functions
	virtual muxer& operator = (const muxer& other);

	//! declare move assignment to move over transfer functions
	virtual muxer& operator = (muxer&& other);



	// >>>>>>>>>>>> ACCESSORS <<<<<<<<<<<<
	
	//! get gradient leaves
	virtual std::unordered_set<ileaf*> get_leaves (void) const;



	// >>>>>>>>>>>> MUTATORS <<<<<<<<<<<<

	//! get tensor data
	virtual tensor* get_tensor (void);

	//! get gradient wrt some node, applies jacobians before evaluting resulting tensor
	//! may call get_gradient
	virtual varptr derive (inode* wrt);

	// >>>>>> CALLED BY OBSERVER TO UPDATE <<<<<<

	//! Inherited from iobserver: update data
	virtual void update (void);

private:
	muxer (std::vector<demuxer*> args, SHAPER_F shaper, 
		VAROP_F op, GLUE_F gluer, std::string label);

	muxer (std::vector<demuxer*> args, std::vector<varptr> slices, 
		SHAPER_F shaper, VAROP_F op, std::shared_ptr<glue_io> gio, std::string label);

	muxer (const muxer& other);

	muxer (muxer&& other);

	// >>>>>> POLYMORPHIC CLONERS <<<<<<

	//! implement clone function
	virtual inode* clone_impl (void) const;

	//! move implementation
	virtual inode* move_impl (void);


	//! copy helper
	void copy_helper (const muxer& other);

	//! move helper
	void move_helper (muxer&& other);

	std::unique_ptr<tensor> data_ = nullptr;

	std::shared_ptr<glue_io> gio_;

	std::vector<varptr> slices_;

	SHAPER_F shaper_;

	VAROP_F op_;
};

}

#endif /* TENNCOR_MUXER_HPP */
