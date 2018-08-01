#include "soil/external.hpp"

DataBucket evaluate (Nodeptr& exit_node)
{
	DataSource src = exit_node->calculate();
	return DataBucket(src.data(), src.type(), exit_node->shape());
}
