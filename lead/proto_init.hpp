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

	std::unique_ptr<clay::Tensor> get (void) const override;

	std::unique_ptr<clay::Tensor> get (clay::Shape shape) const override;

private:
	TensorPb pb_;
};

}

#endif /* LEAD_PB_BUILDER_HPP */