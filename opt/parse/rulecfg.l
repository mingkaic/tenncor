%{
#include <string.h>
#include <stdlib.h>

#include "opt/parse/def.h"
#include "opt/parse/rulecfg_y.h"

%}

%option nodefault
%option noyywrap

%%
[\t\r\a\v\b\n ]					; // spaces

\/\/.*							; // comments

;								{ return STMT_TERM; }

=>								{ return ARROW; }

\(								{ return LPAREN; }

\)								{ return RPAREN; }

,								{ return COMMA; }

=								{ return ASSIGN; }

\[ 								{ return LSB; }

\]								{ return RSB; }

\{ 								{ return LCB; }

\}								{ return RCB; }

:								{ return COLON; }

\.\.							{ return VARIADIC; }

comm							{ return COMMUTATIVE; }

[a-zA-Z\_][a-zA-Z\_]{0,31}		{
	strlcpy(yylval.label, yytext, NSYMBOL);
	return SYMBOL;
}

-?(0|[1-9][0-9]*)(\.\d+)?		{
	yylval.number = atof(yytext);
	return NUMBER;
}

.								{
	fprintf(stderr, "unrecognized token %s", yytext);
}

%%
