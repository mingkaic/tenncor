#include "experimental/opt/parse/def.h"

#ifdef PARSE_DEF_HPP

void list_recursive_free (struct NumList* list)
{
	if (NULL == list)
	{
		return;
	}
	list_recursive_free(list->next_);
	free(list);
}

void subgraph_recursive_free (struct Subgraph* sg)
{
	if (NULL == sg)
	{
		return;
	}
	switch (sg->type_)
	{
		case SCALAR:
			break;
		case ANY:
			free(sg->val_.any_);
			break;
		case BRANCH:
		{
			struct Branch* branch = sg->val_.branch_;
			arglist_recursive_free(branch->args_);
			free(branch);
		}
			break;
	}
	free(sg);
}

void arg_recursive_free (struct Arg* arg)
{
	if (NULL == arg)
	{
		return;
	}
	subgraph_recursive_free(arg->subgraph_);
	list_recursive_free(arg->shaper_);
	list_recursive_free(arg->coorder_);
	free(arg);
}

void arglist_recursive_free (struct ArgList* arglist)
{
	if (NULL == arglist)
	{
		return;
	}
	arglist_recursive_free(arglist->next_);
	arg_recursive_free(arglist->val_);
}

void conversion_recursive_free (struct Conversion* conv)
{
	if (NULL == conv)
	{
		return;
	}
	subgraph_recursive_free(conv->source_);
	subgraph_recursive_free(conv->dest_);
	free(conv);
}

void stmts_recursive_free (struct StmtList* stmts)
{
	if (NULL == stmts)
	{
		return;
	}
	stmts_recursive_free(stmts->next_);
	switch (stmts->type_)
	{
		case SYMBOL_DEF:
		{
			char* str = (char*) stmts->val_;
			free(str);
		}
			break;
		case CONVERSION:
		{
			struct Conversion* conv = (struct Conversion*) stmts->val_;
			conversion_recursive_free(conv);
		}
			break;
	}
	free(stmts);
}

#endif
