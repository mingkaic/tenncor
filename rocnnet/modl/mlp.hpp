#include "prx/api.hpp"

#include "rocnnet/modl/marshal.hpp"

#ifndef MODL_MLP_HPP
#define MODL_MLP_HPP

namespace modl
{

static const std::string weight_fmt = "weight_%d";

static const std::string bias_fmt = "bias_%d";

struct MLP final : public iMarshalSet
{
	MLP (ade::DimT n_input, std::vector<LayerInfo> layers, std::string label) :
		iMarshalSet(label)
	{
		for (size_t i = 0, n = layers.size(); i < n; ++i)
		{
			ade::DimT n_output = layers[i].n_out_;
			ade::Shape weight_shape({n_output, n_input});
			ade::NElemT nweight = weight_shape.n_elems();

			PybindT bound = 1.0 / std::sqrt(n_input);
			std::uniform_real_distribution<PybindT> dist(-bound, bound);
			auto gen = [&dist]()
			{
				return dist(ead::get_engine());
			};
			std::vector<PybindT> wdata(nweight);
			std::generate(wdata.begin(), wdata.end(), gen);

			ead::VarptrT<PybindT> weight = ead::make_variable<PybindT>(
				wdata.data(), weight_shape, fmts::sprintf(weight_fmt, i));

			ead::VarptrT<PybindT> bias = ead::make_variable_scalar<PybindT>(
				0.0, ade::Shape({n_output}), fmts::sprintf(bias_fmt, i));

			layers_.push_back(HiddenLayer{
				std::make_shared<MarshalVar>(weight),
				std::make_shared<MarshalVar>(bias),
				layers[i].hidden_,
			});
			n_input = n_output;
		}
	}

	MLP (const pbm::GraphInfo& graph,
		std::vector<NonLinearF> hiddens, std::string label) :
		iMarshalSet(label)
	{
		const auto& children = graph.tens_.children_;
		size_t nlayers = children.size() / 2;
		if (nlayers == 0)
		{
			logs::fatal("expecting at least one layer");
		}
		size_t nhiddens = hiddens.size();
		if (nlayers != nhiddens)
		{
			logs::warnf("number of layers (%d) do not match "
				"number of nonlinearities specified (%d)", nlayers, nhiddens);
		}
		for (size_t i = 0, n = std::min(nhiddens, nlayers); i < n; ++i)
		{
			std::string weight_label = fmts::sprintf(weight_fmt, i);
			auto wmarsh_it = children.find(weight_label);
			if (children.end() == wmarsh_it)
			{
				logs::fatalf("cannot find weight_%d variable marshaller", i);
			}
			auto weight_it = wmarsh_it->second->tens_.find(weight_label);
			if (wmarsh_it->second->tens_.end() == weight_it)
			{
				logs::fatalf("cannot find weight_%d variable", i);
			}
			std::string bias_label = fmts::sprintf(bias_fmt, i);
			auto bmarsh_it = children.find(bias_label);
			if (children.end() == bmarsh_it)
			{
				logs::fatalf("cannot find bias_%d variable marshaller", i);
			}
			auto bias_it = bmarsh_it->second->tens_.find(bias_label);
			if (bmarsh_it->second->tens_.end() == bias_it)
			{
				logs::fatalf("cannot find bias_%d variable", i);
			}
			layers_.push_back(HiddenLayer{
				std::make_shared<MarshalVar>(
					std::static_pointer_cast<ead::VariableNode<PybindT>>(
					ead::NodeConverters<PybindT>::to_node(weight_it->second))),
				std::make_shared<MarshalVar>(
					std::static_pointer_cast<ead::VariableNode<PybindT>>(
					ead::NodeConverters<PybindT>::to_node(bias_it->second))),
				hiddens[i],
			});
		}
	}

	MLP (const MLP& other) : iMarshalSet(other)
	{
		copy_helper(other);
	}

	MLP& operator = (const MLP& other)
	{
		if (this != &other)
		{
			iMarshalSet::operator = (other);
			copy_helper(other);
		}
		return *this;
	}

	MLP (MLP&& other) = default;

	MLP& operator = (MLP&& other) = default;


	ead::NodeptrT<PybindT> operator () (ead::NodeptrT<PybindT> input)
	{
		ead::NodeptrT<PybindT> out = input;
		for (HiddenLayer& layer : layers_)
		{
			auto hypothesis = prx::fully_connect({out},
				{ead::convert_to_node(layer.weight_->var_)},
				ead::convert_to_node(layer.bias_->var_));
			out = layer.hidden_(hypothesis);
		}
		return out;
	}

	ade::DimT get_ninput (void) const
	{
		return layers_.front().weight_->var_->shape().at(1);
	}

	ade::DimT get_noutput (void) const
	{
		return layers_.back().weight_->var_->shape().at(0);
	}

	MarsarrT get_subs (void) const override
	{
		MarsarrT out;
		out.reserve(2 * layers_.size());
		for (const HiddenLayer& layer : layers_)
		{
			out.push_back(layer.weight_);
			out.push_back(layer.bias_);
		}
		return out;
	}

	struct HiddenLayer
	{
		MarVarsptrT weight_;

		MarVarsptrT bias_;

		NonLinearF hidden_;
	};

	std::vector<HiddenLayer> layers_;

private:
	void copy_helper (const MLP& other)
	{
		layers_.clear();
		for (const HiddenLayer& olayer : other.layers_)
		{
			layers_.push_back(HiddenLayer{
				std::make_shared<MarshalVar>(*olayer.weight_),
				std::make_shared<MarshalVar>(*olayer.bias_),
				olayer.hidden_
			});
		}
	}

	iMarshaler* clone_impl (void) const override
	{
		return new MLP(*this);
	}
};

using MLPptrT = std::shared_ptr<MLP>;

}

#endif // MODL_MLP_HPP
