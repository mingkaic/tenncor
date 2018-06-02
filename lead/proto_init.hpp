/*!
 *
 *  builder.hpp
 *  kiln
 *
 *  Purpose:
 *  abstract builder that validates type and shape
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "clay/ibuilder.hpp"

#include "lead/clay.pb.h"
#include "lead/clay_packer.hpp"

#pragma once
#ifndef LEAD_PB_BUILDER_HPP
#define LEAD_PB_BUILDER_HPP

namespace lead
{

class PbBuilder final : public clay::iBuilder
{
public:
	PbBuilder (const TensorPb& pb);

	clay::TensorPtrT get (void) const override;

	clay::TensorPtrT get (clay::Shape shape) const override;

protected:
	clay::iBuilder* clone_impl (void) const override
	{
		return new PbBuilder(*this);
	}

private:
	TensorPb pb_;
};

}

#endif /* LEAD_PB_BUILDER_HPP */
