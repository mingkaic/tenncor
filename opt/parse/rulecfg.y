
%{
#include <stdio.h>
#include <string.h>

#include "opt/parse/def.h"

extern FILE* yyin;
extern int yyparse();
extern YY_FLUSH_BUFFER;

%}

%start rules

%parse-param {struct PtrList** cversions}

%union
{
	double 				number;
	char 				label[NSYMBOL];

	// objs
	struct Conversion* 	conv;
	struct TreeNode*	node;
	struct Functor*		functor;
	struct Arg*			argument;
	struct KeyVal*		kvpair;

	// lists
	struct NumList* 	nums;
	struct PtrList* 	objs;
}

%type <number> 		NUMBER
%type <label> 		SYMBOL

%type <conv> 		conversion
%type <node>		matcher_el target
%type <functor>		matcher mfunction tfunction
%type <argument> 	marg targ
%type <kvpair>		key_val

%type <objs>		margs targs edge_attr
%type <nums> 		num_arr

%token
STMT_TERM ARROW LPAREN RPAREN COMMA ASSIGN LSB RSB LCB RCB
COLON VARIADIC COMMUTATIVE SYMBOL NUMBER ERROR

%% /* beginning of rules section */

rules:		%empty
			{
				*cversions = new_ptrlist(CONVERSION);
			}
			|
			rules conversion STMT_TERM
			{
				// declare conversion
				ptrlist_pushback(*cversions, $2);
			}

conversion:	matcher ARROW target
			{
				struct Conversion* cv = $$ = malloc(sizeof(struct Conversion));
				cv->matcher_ = $1;
				cv->target_ = $3;
			}

matcher:	mfunction
			{
				$$ = $1;
			}
			|
			COMMUTATIVE mfunction
			{
				$$ = $2;
				$$->commutative_ = TRUE;
			}

matcher_el:	NUMBER
			{
				size_t nbytes = sizeof(struct TreeNode);
				struct TreeNode* node = $$ = malloc(nbytes);
				node->type_ = SCALAR;
				node->val_.scalar_ = $1;
			}
			|
			SYMBOL
			{
				size_t nbytes = sizeof(struct TreeNode);
				struct TreeNode* node = $$ = malloc(nbytes);
				node->type_ = ANY;
				char* dst = node->val_.any_ = malloc(NSYMBOL);
				strncpy(dst, $1, NSYMBOL);
			}
			|
			matcher
			{
				size_t nbytes = sizeof(struct TreeNode);
				struct TreeNode* node = $$ = malloc(nbytes);
				node->type_ = FUNCTOR;
				node->val_.functor_ = $1;
			}

mfunction:	SYMBOL LPAREN margs RPAREN
			{
				size_t nbytes = sizeof(struct Functor);
				struct Functor* f = $$ = malloc(nbytes);
				memset(f, 0, nbytes);

				strncpy(f->name_, $1, NSYMBOL);
				ptrlist_move(&f->args_, $3);
				free($3);
			}
			|
			SYMBOL LPAREN margs COMMA VARIADIC SYMBOL RPAREN
			{
				size_t nbytes = sizeof(struct Functor);
				struct Functor* f = $$ = malloc(nbytes);
				memset(f, 0, nbytes);

				strncpy(f->name_, $1, NSYMBOL);
				strncpy(f->variadic_, $6, NSYMBOL);
				ptrlist_move(&f->args_, $3);
				free($3);
			}

margs:		margs COMMA marg
			{
				struct PtrList* list = $$ = $1;
				ptrlist_pushback(list, $3);
			}
			|
			marg
			{
				struct PtrList* list = $$ = new_ptrlist(ARGUMENT);
				ptrlist_pushback(list, $1);
			}

marg:		matcher_el
			{
				size_t nbytes = sizeof(struct Arg);
				struct Arg* a = $$ = malloc(nbytes);
				memset(a, 0, nbytes);
				a->node_ = $1;
				a->attrs_.type_ = KV_PAIR;
			}
			|
			matcher_el ASSIGN LCB edge_attr RCB
			{
				size_t nbytes = sizeof(struct Arg);
				struct Arg* a = $$ = malloc(nbytes);
				memset(a, 0, nbytes);
				a->node_ = $1;

				ptrlist_move(&a->attrs_, $4);
				free($4);
			}

target:		NUMBER
			{
				size_t nbytes = sizeof(struct TreeNode);
				struct TreeNode* node = $$ = malloc(nbytes);
				node->type_ = SCALAR;
				node->val_.scalar_ = $1;
			}
			|
			SYMBOL
			{
				size_t nbytes = sizeof(struct TreeNode);
				struct TreeNode* node = $$ = malloc(nbytes);
				node->type_ = ANY;
				char* dst = node->val_.any_ = malloc(NSYMBOL);
				strncpy(dst, $1, NSYMBOL);
			}
			|
			tfunction
			{
				size_t nbytes = sizeof(struct TreeNode);
				struct TreeNode* node = $$ = malloc(nbytes);
				node->type_ = FUNCTOR;
				node->val_.functor_ = $1;
			}

tfunction:	SYMBOL LPAREN targs RPAREN
			{
				size_t nbytes = sizeof(struct Functor);
				struct Functor* f = $$ = malloc(nbytes);
				memset(f, 0, nbytes);

				strncpy(f->name_, $1, NSYMBOL);
				ptrlist_move(&f->args_, $3);
				free($3);
			}
			|
			SYMBOL LPAREN targs COMMA VARIADIC SYMBOL RPAREN
			{
				size_t nbytes = sizeof(struct Functor);
				struct Functor* f = $$ = malloc(nbytes);
				memset(f, 0, nbytes);

				strncpy(f->name_, $1, NSYMBOL);
				strncpy(f->variadic_, $6, NSYMBOL);
				ptrlist_move(&f->args_, $3);
				free($3);
			}
			|
			SYMBOL LPAREN VARIADIC SYMBOL RPAREN
			{
				size_t nbytes = sizeof(struct Functor);
				struct Functor* f = $$ = malloc(nbytes);
				memset(f, 0, nbytes);

				strncpy(f->name_, $1, NSYMBOL);
				strncpy(f->variadic_, $4, NSYMBOL);
				f->args_.type_ = ARGUMENT;
			}

targs:		targs COMMA targ
			{
				struct PtrList* list = $$ = $1;
				ptrlist_pushback(list, $3);
			}
			|
			targ
			{
				struct PtrList* list = $$ = new_ptrlist(ARGUMENT);
				ptrlist_pushback(list, $1);
			}

targ:		target
			{
				size_t nbytes = sizeof(struct Arg);
				struct Arg* a = $$ = malloc(nbytes);
				memset(a, 0, nbytes);
				a->node_ = $1;
				a->attrs_.type_ = KV_PAIR;
			}
			|
			target ASSIGN LCB edge_attr RCB
			{
				size_t nbytes = sizeof(struct Arg);
				struct Arg* a = $$ = malloc(nbytes);
				memset(a, 0, nbytes);
				a->node_ = $1;

				ptrlist_move(&a->attrs_, $4);
				free($4);
			}

edge_attr:	edge_attr COMMA key_val
			{
				struct PtrList* list = $$ = $1;
				ptrlist_pushback(list, $3);
			}
			|
			key_val
			{
				struct PtrList* list = $$ = new_ptrlist(KV_PAIR);
				ptrlist_pushback(list, $1);
			}

key_val:	SYMBOL COLON NUMBER
			{
				size_t nbytes = sizeof(struct KeyVal);
				struct KeyVal* kv = $$ = malloc(nbytes);
				memset(kv, 0, nbytes);

				kv->val_scalar_ = TRUE;
				strncpy(kv->key_, $1, NSYMBOL);
				numlist_pushback(&kv->val_, $3);
			}
			|
			SYMBOL COLON LSB num_arr RSB
			{
				size_t nbytes = sizeof(struct KeyVal);
				struct KeyVal* kv = $$ = malloc(nbytes);
				memset(kv, 0, nbytes);

				strncpy(kv->key_, $1, NSYMBOL);
				numlist_move(&kv->val_, $4);
				free($4);
			}
			|
			SYMBOL COLON LSB RSB
			{
				size_t nbytes = sizeof(struct KeyVal);
				struct KeyVal* kv = $$ = malloc(nbytes);
				memset(kv, 0, nbytes);

				strncpy(kv->key_, $1, NSYMBOL);
			}

num_arr:	num_arr COMMA NUMBER
			{
				struct NumList* list = $$ = $1;
				numlist_pushback(list, $3);
			}
			|
			NUMBER
			{
				struct NumList* list = $$ = new_numlist();
				numlist_pushback(list, $1);
			}

%%

int parse_str (struct PtrList** cversions, const char* str)
{
	FILE* tmp = tmpfile();
	if (NULL == tmp)
	{
		puts("Unable to create temp file");
		return 1;
	}
	fputs(str, tmp);
	rewind(tmp);
	return parse_file(cversions, tmp);
}

int parse_file (struct PtrList** cversions, FILE* file)
{
	int exit_status = 1;
	if (file)
	{
		yyin = file;
		YY_FLUSH_BUFFER;
		yyrestart(yyin);
		exit_status = yyparse(cversions);
		fclose(file);
	}
	return exit_status;
}

int yyerror (struct PtrList** cversions, char const* msg)
{
	fprintf(stderr, "%s\n", msg);
	cversions_free(*cversions);
	*cversions = NULL;
	return 1;
}
