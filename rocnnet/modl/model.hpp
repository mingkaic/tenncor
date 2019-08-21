#include "rocnnet/modl/dense.hpp"

#ifndef MODL_MODEL_HPP
#define MODL_MODEL_HPP

namespace modl
{

struct SequentialModel final : public iLayer
{
	SequentialModel (std::string label) :
		iLayer(label) {}

	SequentialModel (const SequentialModel& other) :
		iLayer(other)
	{
		copy_helper(other);
	}

	SequentialModel& operator = (const SequentialModel& other)
	{
		if (this != &other)
		{
			copy_helper(other);
		}
		return *this;
	}

	SequentialModel (SequentialModel&& other) = default;

	SequentialModel& operator = (SequentialModel&& other) = default;

    SequentialModel* clone (void) const
    {
        return static_cast<SequentialModel*>(this->clone_impl());
    }

	ead::NodeptrT<PybindT> connect (ead::NodeptrT<PybindT> input) const override
	{
		auto out = input;
		for (auto& layer : layers_)
		{
			out = layer->connect(out);
		}
		return out;
	}

	void push_back (LayerptrT layer)
	{
		layers_.push_back(layer);
	}

	MarsarrT get_subs (void) const override
	{
		MarsarrT out;
		out.reserve(layers_.size());
		for (auto& layer : layers_)
		{
			auto tmp = layer->get_subs();
			out.insert(out.end(), tmp.begin(), tmp.end());
		}
		return out;
	}

private:
	iMarshaler* clone_impl (void) const override
	{
		return new SequentialModel(*this);
	}

	void copy_helper (const SequentialModel& other)
	{
		layers_.clear();
		layers_.reserve(other.layers_.size());
		std::transform(other.layers_.begin(), other.layers_.end(),
			std::back_inserter(layers_),
			[](LayerptrT layer)
			{
				return LayerptrT(layer->clone());
			});
	}

	std::vector<LayerptrT> layers_;
};

using ModelptrT = std::shared_ptr<SequentialModel>;

ModelptrT load_sequential (const pbm::GraphInfo& graph, std::string label)
{
	return nullptr;
}

}

#endif // MODL_MODEL_HPP
