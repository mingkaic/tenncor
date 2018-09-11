%{
#include <stdio.h>

#include "cli/ade/ast/ast.h"

extern int yylex();
extern int yyerror();
extern FILE *yyin;

%}

%start list

%union
{
	int int_type;
	char str_type[32];
	struct ShapeHolder* vec_type;
	struct ASTNode* ref_type;
}

%type <int_type> UNARY BINARY SHAPEOP INTEGER
%type <str_type> VAR
%type <vec_type> shape
%type <ref_type> expr

%token INTEGER VAR UNARY BINARY SHAPEOP GRAD PRINT EXIT MODE
ASSIGN LPAREN RPAREN LSB RSB COMMA PLUS MINUS STAR SLASH NEWLINE

%left PLUS MINUS
%left STAR SLASH
%left UMINUS /*supplies precedence for unary minus */

%% /* beginning of rules section */

list: 	/* empty */
		|
		list EXIT
		{
			YYACCEPT;
		}
		|
		list stat NEWLINE
		|
		list error NEWLINE
		{
			yyerrok;
		}

stat:	/* empty */
		{}
		|
		MODE VAR
		{
			use_mode($2);
		}
		|
		expr
		{
			show_shape($1);
			free_ast($1);
		}
		|
		VAR ASSIGN expr
		{
			save_ast($1, $3);
			free_ast($3);
		}
		|
		PRINT
		{
			printf("\n");
		}
		|
		PRINT LPAREN expr RPAREN
		{
			show_eq($3);
			free_ast($3);
		}
		;

expr:   GRAD LPAREN expr COMMA VAR RPAREN
		{
			struct ASTNode* node = load_ast($5);
			$$ = grad($3, node);
			free_ast($3);
			free_ast(node);

		}
		|
		SHAPEOP LPAREN expr COMMA shape RPAREN
		{
			$$ = shapeop($3, $5, $1);
			free_ast($3);
			free_shape($5);
		}
		|
		BINARY LPAREN expr COMMA expr RPAREN
		{
			$$ = binary($3, $5, $1);
			free_ast($3);
			free_ast($5);
		}
		|
		UNARY LPAREN expr RPAREN
		{
			$$ = unary($3, $1);
			free_ast($3);
		}
		|
		LPAREN expr RPAREN
		{
			$$ = $2;
		}
		|
		expr STAR expr
		{
			$$ = binary($1, $3, MUL);
			free_ast($1);
			free_ast($3);
		}
		|
		expr SLASH expr
		{
			$$ = binary($1, $3, DIV);
			free_ast($1);
			free_ast($3);
		}
		|
		expr PLUS expr
		{
			$$ = binary($1, $3, ADD);
			free_ast($1);
			free_ast($3);
		}
		|
		expr MINUS expr
		{
			$$ = binary($1, $3, SUB);
			free_ast($1);
			free_ast($3);
		}
		|
		MINUS expr %prec UMINUS
		{
			$$ = unary($2, NEG);
			free_ast($2);
		}
		|
		VAR
		{
			$$ = load_ast($1);
		}
		|
		LSB shape RSB
		{
			$$ = to_node($2);
			free_shape($2);
		}
		|
		LSB RSB
		{
			$$ = empty_node();
		}
		;

shape:	shape COMMA INTEGER
		{
			append($1, $3);
			$$ = $1;
		}
		|
		INTEGER
		{
			$$ = make($1);
		}
		;

%%
int main (int argc, char** argv)
{
	int exit_status = 0;
	if (argc > 1)
	{
		int i;
		for (i = 1; i < argc; ++i)
		{
			FILE* file = fopen(argv[i], "r");
			if (file)
			{
				yyin = file;
				exit_status &= yyparse();
				fclose(file);
			}
			else
			{
				// warn
			}
		}
	}
	else
	{
		exit_status = yyparse();
	}
	return exit_status;
}

int yyerror(s)
char *s;
{
	// todo: improve logger
	fprintf(stderr, "%s\n",s);
}

int yywrap()
{
	return(1);
}
