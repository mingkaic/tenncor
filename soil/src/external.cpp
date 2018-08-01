#include "soil/external.hpp"

DataBucket evaluate (Nodeptr& exit_node)
{
	return DataBucket(
		exit_node->calculate(),
		exit_node->type(),
		exit_node->shape());
}
