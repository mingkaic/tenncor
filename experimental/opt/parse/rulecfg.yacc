
%{
#include <stdio.h>
#include <string.h>

#include "experimental/opt/parse/def.h"

extern FILE* yyin;

%}

%start rules

%parse-param {struct PtrList* stmts}

%union
{
	double dec_type;
	char str_type[32];
	struct NumList* arr_type;
	struct Arg* arg_type;
	struct Subgraph* sg_type;
	struct Statement* stmt_type;
	struct PtrList* ptrs_type;
}

%type <dec_type> NUMBER
%type <str_type> EL
%type <arr_type> num_arr
%type <arg_type> key_val edge_def arg
%type <sg_type> subgraph
%type <ptrs_type> args
%type <stmt_type> conversion group_def symbol_def

%token
SYMBOL NEWLINE ARROW LPAREN RPAREN COMMA ASSIGN LSB RSB
LCB RCB GROUP GROUPDEF SHAPER COORDER COLON EL NUMBER VARIADIC

%% /* beginning of rules section */

rules:		/* empty */
			{
				stmts = new_ptrlist(STATEMENT);
			}
			|
			rules symbol_def
			{
				// declare symbol
				ptrlist_pushback(stmts, $2);
			}
			|
			rules group_def
			{
				// declare group
				ptrlist_pushback(stmts, $2);
			}
			|
			rules conversion
			{
				// declare conversion
				ptrlist_pushback(stmts, $2);
			}

symbol_def: SYMBOL EL NEWLINE
			{
				char* str = malloc(sizeof($2));
				memcpy(str, $2, sizeof($2));
				struct Statement* stmt = yylval.stmt_type =
					malloc(sizeof(struct Statement));
				stmt->type_ = SYMBOL_DEF;
				stmt->val_ = str;
			}

group_def: 	GROUPDEF EL EL NEWLINE
			{
				struct Group* group = malloc(sizeof(struct Group));
				strncpy(group->ref_, $2, 32);
				strncpy(group->tag_, $3, 32);
				struct Statement* stmt = yylval.stmt_type =
					malloc(sizeof(struct Statement));
				stmt->type_ = GROUP_DEF;
				stmt->val_ = group;
			}

conversion:	subgraph ARROW subgraph NEWLINE
			{
				struct Conversion* cv =
					malloc(sizeof(struct Conversion));
				cv->source_ = $1;
				cv->dest_ = $3;
				struct Statement* stmt = yylval.stmt_type =
					malloc(sizeof(struct Statement));
				stmt->type_ = CONVERSION;
				stmt->val_ = cv;
			}

subgraph:	NUMBER
			{
				// a scalar
				struct Subgraph* sg = yylval.sg_type =
					malloc(sizeof(struct Subgraph));
				sg->type_ = SCALAR;
				sg->val_.scalar_ = $1;
			}
			|
			EL
			{
				// a symbol
				struct Subgraph* sg = yylval.sg_type =
					malloc(sizeof(struct Subgraph));
				sg->type_ = ANY;
				char* str = sg->val_.any_ = malloc(sizeof($1));
				memcpy(str, $1, sizeof($1));
			}
			|
			GROUP EL LPAREN args RPAREN
			{
				// a group
				struct Subgraph* sg = yylval.sg_type =
					malloc(sizeof(struct Subgraph));
				sg->type_ = BRANCH;

				size_t nbranchbytes = sizeof(struct Branch);
				struct Branch* branch = sg->val_.branch_ =
					malloc(nbranchbytes);
				memset(branch, 0, nbranchbytes);
				branch->is_group_ = 1;
				branch->args_ = $4;
				strncpy(branch->label_, $2, 32);
			}
			|
			GROUP EL LPAREN args COMMA VARIADIC EL RPAREN
			{
				// a group with variadic
				struct Subgraph* sg = yylval.sg_type =
					malloc(sizeof(struct Subgraph));
				sg->type_ = BRANCH;

				struct Branch* branch = sg->val_.branch_ =
					malloc(sizeof(struct Branch));
				branch->is_group_ = 1;
				branch->args_ = $4;
				strncpy(branch->label_, $2, 32);
				strncpy(branch->variadic_, $7, 32);
			}
			|
			EL LPAREN args RPAREN
			{
				// a functor
				struct Subgraph* sg = yylval.sg_type =
					malloc(sizeof(struct Subgraph));
				sg->type_ = BRANCH;

				size_t nbranchbytes = sizeof(struct Branch);
				struct Branch* branch = sg->val_.branch_ =
					malloc(nbranchbytes);
				memset(branch, 0, nbranchbytes);
				branch->is_group_ = 0;
				branch->args_ = $3;
				strncpy(branch->label_, $1, 32);
			}

args:		args COMMA arg
			{
				struct PtrList* arr = yylval.ptrs_type = $1;
				ptrlist_pushback(arr, $3);
			}
			|
			arg
			{
				struct PtrList* arr = yylval.ptrs_type = new_ptrlist(ARGUMENT);
				ptrlist_pushback(arr, $1);
			}

arg:		subgraph
			{
				// plain old argument
				struct Arg* arg = yylval.arg_type =
					malloc(sizeof(struct Arg));
				memset(arg, 0, sizeof(struct Arg));
				arg->subgraph_ = $1;
			}
			|
			subgraph ASSIGN LCB edge_def RCB
			{
				// argument with edge definition
				struct Arg* arg = yylval.arg_type = $4;
				arg->subgraph_ = $1;
			}

edge_def:	key_val
			{
				yylval.arg_type = $1;
			}
			|
			edge_def COMMA key_val
			{
				struct Arg* arg = yylval.arg_type = $1;
				struct Arg* other = $3;
				if (NULL != other->shaper_)
				{
					arg->shaper_ = other->shaper_;
				}
				if (NULL != other->coorder_)
				{
					arg->coorder_ = other->coorder_;
				}
				arg_recursive_free(other);
			}

key_val:	SHAPER COLON num_arr
			{
				struct Arg* arg = yylval.arg_type =
					malloc(sizeof(struct Arg));
				memset(arg, 0, sizeof(struct Arg));
				arg->shaper_ = $3;
			}
			COORDER COLON num_arr
			{
				struct Arg* arg = yylval.arg_type =
					malloc(sizeof(struct Arg));
				memset(arg, 0, sizeof(struct Arg));
				arg->coorder_ = $3;
			}

num_arr:	num_arr COMMA NUMBER
			{
				struct NumList* arr = yylval.arr_type = $1;
				numlist_pushback(arr, $3);
			}
			|
			NUMBER
			{
				struct NumList* arr = yylval.arr_type = new_numlist();
				numlist_pushback(arr, $1);
			}

%%

int parse_rule (struct PtrList** stmts, const char* filename)
{
	int exit_status = 1;
	FILE* file = fopen(filename, "r");
	if (file)
	{
		yyin = file;
		exit_status = yyparse(*stmts);
		fclose(file);
	}
	return exit_status;
}

int yyerror (struct PtrList* stmts, char const* s)
{
	fprintf(stderr, "%s\n", s);
	return 1;
}
