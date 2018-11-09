#include "adhoc/llo/api.hpp"
#include "adhoc/llo/shear.hpp"

#include "rocnnet/eqns/err_approx.hpp"

DeltasT sgd (llo::DataNode& root, std::vector<llo::DataNode> leaves,
	double learning_rate)
{
	DeltasT errs;
	for (llo::DataNode& leaf : leaves)
	{
		std::shared_ptr<llo::iSource> leafsrc = leaf.source();
		if (nullptr == leafsrc)
		{
			err::warnf("attempting to approximate error of root with respect "
				"to context-less tensor %s... skipping",
				leaf.tensor_->to_string().c_str());
			continue;
		}

		// given root = f, err(x) ~ x - η * df(x), where η is the learning rate
		llo::DataNode gres = llo::zero_prune(root.derive(leaf));
		ade::Shape gshape = gres.tensor_->shape();
		errs.emplace(leafsrc.get(), llo::sub(leaf,
			llo::mul(gres, llo::shaped_scalar(learning_rate, gshape))));
	}
	return errs;
}

DeltasT rms_momentum (llo::DataNode& root,
	std::vector<llo::DataNode> leaves, double learning_rate,
	double discount_factor, double epsilon)
{
	DeltasT errs;
	for (llo::DataNode& leaf : leaves)
	{
		std::shared_ptr<llo::iSource> leafsrc = leaf.source();
		if (nullptr == leafsrc)
		{
			err::warnf("attempting to approximate error of root with respect "
				"to context-less tensor %s... skipping",
				leaf.tensor_->to_string().c_str());
			continue;
		}
		// given root = f, err(x) ~ x - (η * df(x)) / (sqrt(ε + momentum)),
		// where η is the learning rate, and ε is epsilon
		auto gres = llo::zero_prune(root.derive(leaf));

		// upkeep additional hidden variable momentum: starting with value 1
		// given root = f, err(momentum) ~ χ * momentum + (1 - χ) * df(x) ^ 2,
		// where χ is discount_factor
		ade::Shape shape = leaf.tensor_->shape();
		std::vector<double> wun(shape.n_elems(), 1);
		llo::DataNode momentum = llo::Source<double>::get(shape, wun);
		auto discount_node = llo::shaped_scalar(discount_factor, shape);
		auto datcount_node = llo::shaped_scalar(1.0 - discount_factor, shape);
		errs.insert({momentum.source().get(),
			llo::add(llo::mul(discount_node, momentum),
			llo::prod({datcount_node, gres, gres}))});

		errs.emplace(leafsrc.get(), llo::sub(leaf,
			llo::div(llo::mul(gres, llo::shaped_scalar(learning_rate, shape)),
			llo::add(llo::sqrt(momentum), llo::shaped_scalar(epsilon, shape)))
		));
	}
	return errs;
}
