%{
#include <stdio.h>
#include <string.h>
#include "arbToPtx.cpp"
using namespace std;
    
    
// stuff from lex that yacc needs to know about:
extern int yylex();
extern int yyparse();
extern FILE *yyin;
void yyerror(string s);

extern arbToPtxConverter globalGLSLtoPTXObject;

%}

%error-verbose
//%define parse.lac
// since there's a different C datatype for each of the token type, yacc needs a
// union for the lex return value:
%union {
	int integer_value;
	float float_value;
	char *string_value;
}
// and then you just associate one of the defined token types with one of
// the union fields and we're happy:
//varModifier
%token <integer_value> intConst
%token <float_value> floatConst
%token <string_value> STRING
//varMods
%token SHORT LONG INT UINT FLOAT

%token END OPTION
//special instructions
%token KILL DDX DDY
//VECTORop
%token ABS CEIL FLR FRC I2F LIT MOV MOV_SAT NOT NRM PK2H PK2US PK4B PK4UB ROUND SSG TRUNC
//SCALARopf
%token COS EX2 LG2 RCC RCP RSQ SCS SIN UP2H UP2US UP4B UP4UB
//BINSCop
%token POW
//VECSCAop              
%token DIV SHL SHR MOD
//BINop
%token ADD AND DP3 DP4 DPH DST MAX MIN MUL OR RFL SEQ SFL SGE SGT SLE SLT SNE STR SUB XPD DP2 XOR
//TRIop
%token CMP DP2A LRP MAD SAD X2D
//SWZop
%token SWZ
//TEXop
%token TEX TXB TXF TXL TXP TXQ
//TXDop
%token TXD
//BRAop
%token CAL
//FLOWCCop
%token RET BRK CONT
//IFop
%token IF
//REPop
%token REP
//ENDFLOWop
%token ELSE ENDIF ENDREP
//opModifier
%token CC CC0 CC1 SAT SSAT NTC S24 U24 HI
//texture
%token texture 
//texTarget
%token oneD twoD threeD CUBE RECT SHADOW1D SHADOW2D SHADOWRECT ARRAY1D ARRAY2D SHADOWCUBE SHADOWARRAY1D SHADOWARRAY2D
//misc
%token ATTRIB PARAM TEMP OUTPUT ALIAS
//interpModifier
%token FLAT CENTROID NOPERSPECTIVE
//bufferDeclType
%token BUFFER BUFFER4
//attribBasic
%token fogcoord facing primitive ID position weight normal  instance color texcoord clip attrib fragment vertex
//state had to be capital to not conflict with some stdlib var
%token STATE material ambient diffuse specular emission shininess light attenuation spot half direction lightmodel scenecolor lightprod fog params row matrix inverse transpose invtrans modelview projection mvp texgen object eye program plane point SIZE texenv range env local depth buffer pointsize result TWO_DOTS
//ccMaskRule
%token EQ GE GT LE LT NE TR FL EQ0 GE0 GT0 LE0 LT0 NE0 TR0 FL0 EQ1 GE1 GT1 LE1 LT1 NE1 TR1 FL1 NAN NAN0 NAN1 LEG LEG0 LEG1 CF CF0 CF1 NCF NCF0 NCF1 OF OF0 OF1 NOF NOF0 NOF1 AB AB0 AB1 BLE BLE0 BLE1 SF SF0 SF1 NSF NSF0 NSF1
%token XY XZ YZ XYZ XW YW XYW ZW XZW YZW XYZW
%token RG RB GB RGB RA GA RGA BA RBA GBA RGBA
%token front back primary secondary
%token<string_value> xyzwSwizzleMask rgbaSwizzleMask
//to indicate if we are dealing with vertex or fragment program
%token VERTEX_PROGRAM FRAGMENT_PROGRAM
%%

// the first rule defined is the highest-level rule, which in our 
// case is just the concept of a whole ARB program:
FULL_PROGRAM           : program_type optionSequence declSequence 
                            statementSequence END       {globalGLSLtoPTXObject.set_endOfProgram();}
                        ;

program_type           : VERTEX_PROGRAM         {globalGLSLtoPTXObject.set_program_type("VERTEX_PROGRAM");}
                            | FRAGMENT_PROGRAM  {globalGLSLtoPTXObject.set_program_type("FRAGMENT_PROGRAM");}
                        ;
                        
optionSequence        : option optionSequence
                            | /* empty */
                        ;

option                : OPTION identifier ';'
                        ;

declSequence          : /* empty */
                        ;

statementSequence     : statement {globalGLSLtoPTXObject.set_statementFound();} statementSequence
                            | /* empty */
                        ;

statement             : instruction ';'                 {globalGLSLtoPTXObject.set_statement("instruction");}
                            | namingStatement ';'       {globalGLSLtoPTXObject.set_statement("namingStatement");}
                            | instLabel ':'
                        ;

instruction           : ALUInstruction          {globalGLSLtoPTXObject.set_instruction("ALUInstruction");}
                            | TexInstruction    {globalGLSLtoPTXObject.set_instruction("TexInstruction");}
                            | FlowInstruction
                            | SpecialInstruction
                        ;

ALUInstruction        : VECTORop_instruction            {globalGLSLtoPTXObject.set_ALUInstruction("VECTORop_instruction");}
                            | SCALARop_instruction      {globalGLSLtoPTXObject.set_ALUInstruction("SCALARop_instruction");}
                            | BINSCop_instruction
                            | BINop_instruction         {globalGLSLtoPTXObject.set_ALUInstruction("BINop_instruction");}
                            | VECSCAop_instruction
                            | TRIop_instruction         {globalGLSLtoPTXObject.set_ALUInstruction("TRIop_instruction");}
                            | SWZop_instruction
                        ;

TexInstruction        : TEXop_instruction               {globalGLSLtoPTXObject.set_TexInstruction("TEXop_instruction");}    
                            | TXDop_instruction         {cout<<"TXDop_instruction is not supported \n"<<endl;abort();}
                        ;

FlowInstruction       : BRAop_instruction
                            | FLOWCCop_instruction
                            | IFop_instruction
                            | REPop_instruction
                            | ENDFLOWop_instruction
                        ;

SpecialInstruction    : KILL opModifiers killCond
                            | DDX opModifiers instResult ',' instOperandV
                            | DDY opModifiers instResult ',' instOperandV
                        ;

killCond	      : instOperandV
			;

VECTORop_instruction  : VECTORop opModifiers instResult {globalGLSLtoPTXObject.add_instResult();}
                            ',' instOperandV            {globalGLSLtoPTXObject.add_instOperand("instOperandV");}
                        ;

VECTORop              : ABS            {globalGLSLtoPTXObject.set_VECTORop("ABS",false);}
                            | CEIL
                            | FLR      {globalGLSLtoPTXObject.set_VECTORop("FLR",false);}
                            | FRC
                            | I2F
                            | LIT
                            | MOV       {globalGLSLtoPTXObject.set_VECTORop("MOV",false);}
                            | MOV_SAT   {globalGLSLtoPTXObject.set_VECTORop("MOV",true);}
                            | NOT
                            | NRM
                            | PK2H
                            | PK2US
                            | PK4B
                            | PK4UB
                            | ROUND
                            | SSG
                            | TRUNC
                        ;

SCALARop_instruction  : SCALARop opModifiers instResult {globalGLSLtoPTXObject.add_instResult();}
                            ',' instOperandV            {globalGLSLtoPTXObject.add_instOperand("instOperandV");} //operand originally scalar zzz
                        ;

SCALARop              : COS
                            | EX2
                            | LG2
                            | RCC
                            | RCP
                            | RSQ       {globalGLSLtoPTXObject.set_SCALARop("RSQ");}
                            | SCS
                            | SIN
                            | UP2H
                            | UP2US
                            | UP4B
                            | UP4UB
                        ;

BINSCop_instruction   : BINSCop opModifiers instResult ',' 
                            instOperandS ',' instOperandS
                        ;

BINSCop               : POW
                        ;

VECSCAop_instruction  : VECSCAop opModifiers instResult ',' 
                            instOperandV ',' instOperandS
                        ;

VECSCAop              : DIV
                            | SHL
                            | SHR
                            | MOD
                        ;

BINop_instruction     : BINop opModifiers instResult    {globalGLSLtoPTXObject.add_instResult();} 
                        ',' instOperandV                {globalGLSLtoPTXObject.add_instOperand("instOperandV");}
                        ',' instOperandV                {globalGLSLtoPTXObject.add_instOperand("instOperandV");}
                        ;

BINop                 : ADD             {globalGLSLtoPTXObject.set_BINop("ADD");}
                            | AND
                            | DP3       {globalGLSLtoPTXObject.set_BINop("DP3");}    
                            | DP4       {globalGLSLtoPTXObject.set_BINop("DP4");} 
                            | DPH
                            | DST
                            | MAX       {globalGLSLtoPTXObject.set_BINop("MAX");}
                            | MIN
                            | MUL       {globalGLSLtoPTXObject.set_BINop("MUL");}
                            | OR
                            | RFL
                            | SEQ       {globalGLSLtoPTXObject.set_BINop("SEQ");}
                            | SFL       
                            | SGE       {globalGLSLtoPTXObject.set_BINop("SGE");}
                            | SGT       {globalGLSLtoPTXObject.set_BINop("SGT");}
                            | SLE       {globalGLSLtoPTXObject.set_BINop("SLE");}
                            | SLT       {globalGLSLtoPTXObject.set_BINop("SLT");}
                            | SNE       {globalGLSLtoPTXObject.set_BINop("SNE");}  
                            | STR
                            | SUB
                            | XPD
                            | DP2
                            | XOR
                        ;

TRIop_instruction     : TRIop opModifiers instResult    {globalGLSLtoPTXObject.add_instResult();}
                        ',' instOperandV                {globalGLSLtoPTXObject.add_instOperand("instOperandV");} 
                        ',' instOperandV                {globalGLSLtoPTXObject.add_instOperand("instOperandV");} 
                        ',' instOperandV                {globalGLSLtoPTXObject.add_instOperand("instOperandV");}
                        ;

TRIop                 : CMP             {globalGLSLtoPTXObject.set_TRIop("CMP");}
                            | DP2A
                            | LRP
                            | MAD       {globalGLSLtoPTXObject.set_TRIop("MAD");}
                            | SAD
                            | X2D
                        ;

SWZop_instruction     : SWZop opModifiers instResult ',' 
                            instOperandVNS ',' extendedSwizzle
                        ;

SWZop                 : SWZ
                        ;

TEXop_instruction     : TEXop opModifiers instResult    {globalGLSLtoPTXObject.add_instResult();}
                        ',' instOperandV                {globalGLSLtoPTXObject.add_instOperand("instOperandV");} 
                        ',' texAccess
                        ;

TEXop                 : TEX             {globalGLSLtoPTXObject.set_TEXop("TEX");}
                            | TXB       {cout<<"TXB is not supported"<<endl;abort();}
                            | TXF       {cout<<"TXF is not supported"<<endl;abort();}
                            | TXL       {cout<<"TXL is not supported"<<endl;abort();}
                            | TXP       {cout<<"TXP is not supported"<<endl;abort();}
                            | TXQ       {cout<<"TXQ is not supported"<<endl;abort();}
                        ;

TXDop_instruction     : TXDop opModifiers instResult ',' 
                            instOperandV ',' instOperandV ',' 
                            instOperandV ',' texAccess
                        ;

TXDop                 : TXD
                        ;

BRAop_instruction     : BRAop opModifiers instTarget 
                            optBranchCond
                        ;

BRAop                 : CAL
                        ;

FLOWCCop_instruction  : FLOWCCop opModifiers optBranchCond
                        ;

FLOWCCop              : RET
                            | BRK
                            | CONT
                        ;

IFop_instruction      : IFop opModifiers ccTest
                        ;

IFop                  : IF
                        ;

REPop_instruction     : REPop opModifiers instOperandV
                            | REPop opModifiers
                        ;

REPop                 : REP
                        ;

ENDFLOWop_instruction : ENDFLOWop opModifiers
                        ;

ENDFLOWop             : ELSE
                            | ENDIF
                            | ENDREP
                        ;

opModifiers           : opModifierItem opModifiers      
                            | /* empty */               
                        ;

opModifierItem        : '.' opModifier  {globalGLSLtoPTXObject.set_opModifierItem();}
                        ;

opModifier            : 'F'             {globalGLSLtoPTXObject.set_opModifier("F");}
                            | 'U'       {globalGLSLtoPTXObject.set_opModifier("U");}               
                            | 'S'       {globalGLSLtoPTXObject.set_opModifier("S");}
                            | CC        {cout<<"CC "<<"is not supported\n"; abort();}
                            | CC0       {cout<<"CC0 "<<"is not supported\n"; abort();}
                            | CC1       {cout<<"CC1 "<<"is not supported\n"; abort();}
                            | SAT       {cout<<"SAT "<<"is not supported\n"; abort();}
                            | SSAT      {cout<<"SSAT "<<"is not supported\n"; abort();}
                            | NTC       {cout<<"NTC "<<"is not supported\n"; abort();}
                            | S24       {cout<<"S24 "<<"is not supported\n"; abort();}
                            | U24       {cout<<"U24 "<<"is not supported\n"; abort();}
                            | HI        {cout<<"HI "<<"is not supported\n"; abort();}
                        ;

texAccess             : texImageUnit ',' texTarget optTexOffset
                        ;

texImageUnit          : texture optArrayMemAbs  {globalGLSLtoPTXObject.set_texImageUnit();}
                        ;

texTarget             : oneD                    {globalGLSLtoPTXObject.set_texTarget("1d");}
                            | twoD              {globalGLSLtoPTXObject.set_texTarget("2d");}
                            | threeD            {globalGLSLtoPTXObject.set_texTarget("3d");}
                            | CUBE              {cout<<"CUBE is not supported\n";abort();}
                            | RECT              {cout<<"RECT is not supported\n";abort();}
                            | SHADOW1D          {cout<<"SHADOW1D is not supported\n";abort();}
                            | SHADOW2D          {cout<<"SHADOW2D is not supported\n";abort();}
                            | SHADOWRECT        {cout<<"SHADOWRECT is not supported\n";abort();}
                            | ARRAY1D           {cout<<"ARRAY1D is not supported\n";abort();}
                            | ARRAY2D           {cout<<"ARRAY2D is not supported\n";abort();}
                            | SHADOWCUBE        {cout<<"SHADOWCUBE is not supported\n";abort();}
                            | SHADOWARRAY1D     {cout<<"SHADOWARRAY1D is not supported\n";abort();}
                            | SHADOWARRAY2D     {cout<<"SHADOWARRAY2D is not supported\n";abort();}
                        ;

optTexOffset          : /*empty*/
                        | ',' texOffset {cout<<"texOffset is not supported \n";abort();}
                        ;

texOffset             : '(' texOffsetComp ')'
                            | '(' texOffsetComp ',' texOffsetComp ')'
                            | '(' texOffsetComp ',' texOffsetComp ',' 
                            texOffsetComp ')'
                        ;

texOffsetComp         : optSign intConst
                        ;

optBranchCond         : /* empty */
                            | ccMask
                        ;

instOperandV          : instOperandAbsV         {cout<<"instOperandAbsV is not supported "<<endl; abort();}
                            | instOperandBaseV  {globalGLSLtoPTXObject.set_instOperandV("instOperandBaseV");}
                        ;

instOperandAbsV       : operandAbsNeg '|' instOperandBaseV '|'
                        ;

instOperandBaseV      : operandNeg attribUseV                          {globalGLSLtoPTXObject.set_instOperandBaseV("attribUseV");}
                            | operandNeg tempAttribParamBufferUseV     {globalGLSLtoPTXObject.set_instOperandBaseV("tempAttribParamBufferUseV");}    
                        ;

instOperandS          : instOperandAbsS         {cout<<"instOperandAbsS is not supported "<<endl; abort();}
                            | instOperandBaseS
                        ;

instOperandAbsS       : operandAbsNeg '|' instOperandBaseS '|'
                        ;

instOperandBaseS      : operandNeg attribUseS
                            | operandNeg tempUseS
                            | operandNeg paramUseS
                            | operandNeg bufferUseS
                        ;

instOperandVNS        : attribUseVNS
                            | tempUseVNS
                            | paramUseVNS
                            | bufferUseVNS
                        ;

operandAbsNeg         : optSign
                        ;

operandNeg            : optSign
                        ;

instResult            : instResultBase optWriteMask optCcMask      {globalGLSLtoPTXObject.set_instResult("instResultBase");}
                        ;

instResultBase        : resultBasic colorType               {globalGLSLtoPTXObject.set_optColorType(true);globalGLSLtoPTXObject.set_optFaceType(false);globalGLSLtoPTXObject.set_instResultBase("resultBasic");}
                        | resultBasic faceType              {globalGLSLtoPTXObject.set_optColorType(false);globalGLSLtoPTXObject.set_optFaceType(true);globalGLSLtoPTXObject.set_instResultBase("resultBasic");}
                        | resultBasic                       {globalGLSLtoPTXObject.set_optColorType(false);globalGLSLtoPTXObject.set_optFaceType(false);globalGLSLtoPTXObject.set_instResultBase("resultBasic");}
                        | resultBasic faceType colorType    {globalGLSLtoPTXObject.set_optColorType(true);globalGLSLtoPTXObject.set_optFaceType(true);globalGLSLtoPTXObject.set_instResultBase("resultBasic");}
                        | resultVarName optArrayMem         {globalGLSLtoPTXObject.set_instResultBase("resultVarName");}                
                        ;                     
                        
namingStatement       : varMods ATTRIB_statement                {globalGLSLtoPTXObject.set_namingStatement("ATTRIB_statement");}
                            | varMods PARAM_statement           {globalGLSLtoPTXObject.set_namingStatement("PARAM_statement");}
                            | varMods TEMP_statement            {globalGLSLtoPTXObject.set_namingStatement("TEMP_statement");}
                            | varMods OUTPUT_statement          {globalGLSLtoPTXObject.set_namingStatement("OUTPUT_statement");}
                            | varMods BUFFER_statement          {cout<<"BUFFER is not supported\n"; abort();}
                            | ALIAS_statement                   {cout<<"ALIAS is not supported\n"; abort();}
                        ;

ATTRIB_statement      : ATTRIB establishName '[' ']' '=' '{' attribUseD '}'
                        ;

PARAM_statement       : PARAM_singleStmt                {globalGLSLtoPTXObject.set_PARAM_statement("PARAM_singleStmt");}
                            | PARAM_multipleStmt        {globalGLSLtoPTXObject.set_PARAM_statement("PARAM_multipleStmt");}
                        ;

PARAM_singleStmt      : PARAM establishName paramSingleInit
                        ;

PARAM_multipleStmt    : PARAM establishName optArraySize paramMultipleInit
                        ;

paramSingleInit       : '=' paramUseDB                          {globalGLSLtoPTXObject.add_paramMultInitList("paramUseDB");}
                        ;

paramMultipleInit     : '=' '{' paramMultInitList '}'
                        ;

paramMultInitList     :       paramUseDM                        {globalGLSLtoPTXObject.add_paramMultInitList("paramUseDM");}
                            | paramUseDM                        {globalGLSLtoPTXObject.add_paramMultInitList("paramUseDM");}  
                              ',' paramMultInitList  
                        ;

TEMP_statement        : TEMP varNameList        
                        ;

OUTPUT_statement      : OUTPUT establishName '=' resultUseD
                            | OUTPUT establishName '[' ']' '=' '{' resultUseD '}'
                        ;

//even the original grammer accepts multi varModifier we here only accept one or none
varMods               : varModifier             {globalGLSLtoPTXObject.set_varMods(true);}
                            | /* empty */       {globalGLSLtoPTXObject.set_varMods(false);}
                        ;

varModifier           : SHORT                   {globalGLSLtoPTXObject.set_varModifier("SHORT");}
                            | LONG              {globalGLSLtoPTXObject.set_varModifier("LONG");}
                            | INT               {globalGLSLtoPTXObject.set_varModifier("INT");}
                            | UINT              {globalGLSLtoPTXObject.set_varModifier("UINT");}
                            | FLOAT             {globalGLSLtoPTXObject.set_varModifier("FLOAT");}
                            | interpModifier    {cout<<"interpModifier is not supported\n"; abort();}
                        ;

interpModifier	      : FLAT
			| CENTROID
			| NOPERSPECTIVE
			;

ALIAS_statement       : ALIAS establishName '=' establishedName
                        ;

BUFFER_statement      : bufferDeclType establishName '=' 
                            bufferSingleInit
                            | bufferDeclType establishName 
                            optArraySize '=' bufferMultInit
                        ;

bufferDeclType        : BUFFER
                            | BUFFER4
                        ;

bufferSingleInit      : '=' bufferUseDB
                        ;

bufferMultInit        : '=' '{' bufferMultInitList '}'
                        ;

bufferMultInitList    : bufferUseDM
                            | bufferUseDM ',' bufferMultInitList
                        ;

varNameList           : establishName           {globalGLSLtoPTXObject.set_varNameList();}
                            | establishName     {globalGLSLtoPTXObject.set_varNameList();} 
                            ',' varNameList
                        ;

attribUseV            : attribBasic swizzleSuffix                               {globalGLSLtoPTXObject.set_attribUseV("attribBasic");}
                            | attribColor '.' colorType swizzleSuffix           {globalGLSLtoPTXObject.set_optColorType(true);globalGLSLtoPTXObject.set_attribUseV("attribColor");}
                            | attribColor swizzleSuffix                         {globalGLSLtoPTXObject.set_optColorType(false);globalGLSLtoPTXObject.set_attribUseV("attribColor");}
                        ;

attribUseS            : attribBasic scalarSuffix
                            | attribVarName scalarSuffix
                            | attribVarName arrayMem scalarSuffix
                            | attribColor scalarSuffix
                            | attribColor '.' colorType scalarSuffix
                        ;

attribUseVNS          : attribBasic
                            | attribVarName
                            | attribVarName arrayMem
                            | attribColor
                            | attribColor '.' colorType
                        ;

attribUseD            : attribBasic                            {globalGLSLtoPTXObject.set_attribUseD("attribBasic");}
                            | attribColor optColorType         {globalGLSLtoPTXObject.set_attribUseD("attribColor");}
                            | attribMulti                      {globalGLSLtoPTXObject.set_attribUseD("attribMulti");}
                        ;

attribBasic            :      fragPrefix fogcoord               {globalGLSLtoPTXObject.set_attribBasic("fragment","fogcoord");}
			    | fragPrefix position               {globalGLSLtoPTXObject.set_attribBasic("fragment","position");}
			    | fragPrefix facing                 {globalGLSLtoPTXObject.set_attribBasic("fragment","facing");}
			    | attribTexCoord optArrayMemAbs     {globalGLSLtoPTXObject.set_attribBasic("attribTexCoord","");}
			    | attribClip arrayMemAbs            {globalGLSLtoPTXObject.set_attribBasic("attribClip","");}
			    | attribGeneric arrayMemAbs         {globalGLSLtoPTXObject.set_attribBasic("attribGeneric","");}
			    | primitive '.' ID                  {globalGLSLtoPTXObject.set_attribBasic("primitive","ID");}
			    | vtxPrefix position                {globalGLSLtoPTXObject.set_attribBasic("vertex","position");}
                            | vtxPrefix weight optArrayMemAbs   {globalGLSLtoPTXObject.set_attribBasic("vertex","weight");}
                            | vtxPrefix normal                  {globalGLSLtoPTXObject.set_attribBasic("vertex","normal");}
                            | vtxPrefix fogcoord                {globalGLSLtoPTXObject.set_attribBasic("vertex","fogcoord");}
                            | vtxPrefix ID                      {globalGLSLtoPTXObject.set_attribBasic("vertex","ID");}
                            | vtxPrefix instance                {globalGLSLtoPTXObject.set_attribBasic("vertex","instance");}
			;

attribColor            : fragPrefix color               {globalGLSLtoPTXObject.set_attribColor("fragment");}
                            | vtxPrefix color           {globalGLSLtoPTXObject.set_attribColor("vertex");}
			;

attribMulti            : attribTexCoord arrayRange      {globalGLSLtoPTXObject.set_attribMulti("attribTexCoord");}
                            | attribClip arrayRange     {globalGLSLtoPTXObject.set_attribMulti("attribClip");}
                            | attribGeneric arrayRange  {globalGLSLtoPTXObject.set_attribMulti("attribGeneric");}
			;

attribTexCoord         : fragPrefix texcoord            {globalGLSLtoPTXObject.set_attribTexCoord("fragment");}
			    | vtxPrefix texcoord        {globalGLSLtoPTXObject.set_attribTexCoord("vertex");}
			;

attribClip             : fragPrefix clip
			;

attribGeneric          : fragPrefix attrib              {globalGLSLtoPTXObject.set_attribGeneric("fragment");}
			    | vtxPrefix attrib          {globalGLSLtoPTXObject.set_attribGeneric("vertex");}
			;

fragPrefix             : fragment '.'
			;

vtxPrefix              :  vertex '.'
			;

tempAttribParamBufferUseV      : paramVarName optArrayMem swizzleSuffix         {globalGLSLtoPTXObject.set_tempAttribParamBufferUseV("paramVarName");}
                            | stateSingleItem swizzleSuffix                     {globalGLSLtoPTXObject.set_tempAttribParamBufferUseV("stateSingleItem");}
                            | programSingleItem swizzleSuffix                   {globalGLSLtoPTXObject.set_tempAttribParamBufferUseV("programSingleItem");}   
                            | constantVector swizzleSuffix                      {globalGLSLtoPTXObject.set_tempAttribParamBufferUseV("constantVector");}
                            | constantScalar                                    {globalGLSLtoPTXObject.set_tempAttribParamBufferUseV("signedConstantScalar");}
                        ;

paramUseS             : paramVarName optArrayMem scalarSuffix
                            | stateSingleItem scalarSuffix
                            | programSingleItem scalarSuffix
                            | constantVector scalarSuffix
                            | constantScalar
                        ;

paramUseVNS           : paramVarName optArrayMem
                            | stateSingleItem
                            | programSingleItem
                            | constantVector
                            | constantScalar
                        ;

paramUseDB            : stateSingleItem                         {globalGLSLtoPTXObject.set_paramUseDB("stateSingleItem");}
                            | programSingleItem                 {globalGLSLtoPTXObject.set_paramUseDB("programSingleItem");}
                            | constantVector                    {globalGLSLtoPTXObject.set_paramUseDB("constantVector");}
                            | signedConstantScalar              {globalGLSLtoPTXObject.set_paramUseDB("signedConstantScalar");}
                        ;

paramUseDM            : stateSingleItem                         {globalGLSLtoPTXObject.set_paramUseDM("stateSingleItem");}
                            | programMultipleItem               {globalGLSLtoPTXObject.set_paramUseDM("programMultipleItem");}
                            | constantVector                    {globalGLSLtoPTXObject.set_paramUseDM("constantVector");}
                            | signedConstantScalar              {globalGLSLtoPTXObject.set_paramUseDM("signedConstantScalar");}
                            | STATE '.' stateMatrixRows         {globalGLSLtoPTXObject.set_paramUseDM("stateMatrixRows");}
                        ;

stateSingleItem       : STATE '.' stateMaterialItem             {globalGLSLtoPTXObject.set_stateSingleItem("stateMaterialItem");}
                            | STATE '.' stateLightItem          {globalGLSLtoPTXObject.set_stateSingleItem("stateLightItem");}
                            | STATE '.' stateLightModelItem     {globalGLSLtoPTXObject.set_stateSingleItem("stateLightModelItem");}
                            | STATE '.' stateLightProdItem      {globalGLSLtoPTXObject.set_stateSingleItem("stateLightProdItem");}
                            | STATE '.' stateFogItem            {globalGLSLtoPTXObject.set_stateSingleItem("stateFogItem");}
                            | STATE '.' stateTexGenItem         {globalGLSLtoPTXObject.set_stateSingleItem("stateTexGenItem");}
                            | STATE '.' stateClipPlaneItem      {globalGLSLtoPTXObject.set_stateSingleItem("stateClipPlaneItem");}
                            | STATE '.' statePointItem          {globalGLSLtoPTXObject.set_stateSingleItem("statePointItem");}
                            | STATE '.' stateTexEnvItem         {globalGLSLtoPTXObject.set_stateSingleItem("stateTexEnvItem");}
                            | STATE '.' stateDepthItem          {globalGLSLtoPTXObject.set_stateSingleItem("stateDepthItem");}
                            | STATE '.' stateMatrixRow          {globalGLSLtoPTXObject.set_stateSingleItem("stateMatrixRow");}
                        ;

stateMaterialItem     : material optFaceType '.' stateMatProperty
                        ;

stateMatProperty      : ambient         {globalGLSLtoPTXObject.set_stateMatProperty("ambient");}
                            | diffuse   {globalGLSLtoPTXObject.set_stateMatProperty("diffuse");}
                            | specular  {globalGLSLtoPTXObject.set_stateMatProperty("specular");}
                            | emission  {globalGLSLtoPTXObject.set_stateMatProperty("emission");}
                            | shininess {globalGLSLtoPTXObject.set_stateMatProperty("shininess");}
                        ;

stateLightItem        : light arrayMemAbs '.' stateLightProperty
                        ;

stateLightProperty    : ambient                         {globalGLSLtoPTXObject.set_stateLightProperty("ambient");}
                            | diffuse                   {globalGLSLtoPTXObject.set_stateLightProperty("diffuse");}
                            | specular                  {globalGLSLtoPTXObject.set_stateLightProperty("specular");}
                            | position                  {globalGLSLtoPTXObject.set_stateLightProperty("position");}
                            | attenuation               {globalGLSLtoPTXObject.set_stateLightProperty("attenuation");}
                            | spot '.' direction        {globalGLSLtoPTXObject.set_stateLightProperty("spot");}
                            | half                      {globalGLSLtoPTXObject.set_stateLightProperty("half");}
                        ;

stateLightModelItem   : lightmodel '.' stateLModProperty
                        ;

stateLModProperty     : ambient                         {globalGLSLtoPTXObject.set_stateLModProperty("ambient");}
                            | scenecolor                {globalGLSLtoPTXObject.set_stateLModProperty("scenecolor");}        
                            | faceType '.' scenecolor   {globalGLSLtoPTXObject.set_stateLModProperty("faceType");}
                        ;

stateLightProdItem    : lightprod arrayMemAbs optFaceType '.' stateLProdProperty
                        ;

stateLProdProperty    : ambient                         {globalGLSLtoPTXObject.set_stateLProdProperty("ambient");}
                            | diffuse                   {globalGLSLtoPTXObject.set_stateLProdProperty("diffuse");}
                            | specular                  {globalGLSLtoPTXObject.set_stateLProdProperty("specular");}
                        ;

stateFogItem          : fog '.' stateFogProperty
                        ;

stateFogProperty      : color           {globalGLSLtoPTXObject.set_stateFogProperty("color");}
                            | params    {globalGLSLtoPTXObject.set_stateFogProperty("params");}
                        ;

stateMatrixRow        : stateMatrixItem '.' stateMatModifier  '.' row arrayMemAbs       {globalGLSLtoPTXObject.set_optStateMatModifier(true);}
                            | stateMatrixItem '.' row arrayMemAbs                       {globalGLSLtoPTXObject.set_optStateMatModifier(false);}
                        ;                        
                        
stateMatrixRows       : stateMatrixItem optRowArrayRange                        {globalGLSLtoPTXObject.set_optStateMatModifier(false);}
                        | stateMatrixItem '.' stateMatModifier optRowArrayRange {globalGLSLtoPTXObject.set_optStateMatModifier(true);}
                        ;

stateMatrixItem       : matrix '.' stateMatrixName
                        ;     
                        
stateMatModifier      : inverse         {globalGLSLtoPTXObject.set_stateMatModifier("inverse");}
                            | transpose {globalGLSLtoPTXObject.set_stateMatModifier("transpose");}
                            | invtrans  {globalGLSLtoPTXObject.set_stateMatModifier("invtrans");}
                        ;

stateMatrixName       : modelview optArrayMemAbs        {globalGLSLtoPTXObject.set_stateMatrixName("modelview");}
                            | projection                {globalGLSLtoPTXObject.set_stateMatrixName("projection");}
                            | mvp                       {globalGLSLtoPTXObject.set_stateMatrixName("mvp");}
                            | texture optArrayMemAbs    {globalGLSLtoPTXObject.set_stateMatrixName("texture");}
                            | program arrayMemAbs       {globalGLSLtoPTXObject.set_stateMatrixName("program");}
                        ;

stateTexGenItem       : texgen optArrayMemAbs '.' stateTexGenType '.' stateTexGenCoord
                        ;

stateTexGenType       : eye             {globalGLSLtoPTXObject.set_stateTexGenType("eye");}
                            | object    {globalGLSLtoPTXObject.set_stateTexGenType("object");}
                        ;

stateTexGenCoord      : 's'             {globalGLSLtoPTXObject.set_stateTexGenCoord("s");}
                            | 't'       {globalGLSLtoPTXObject.set_stateTexGenCoord("t");}
                            | 'r'       {globalGLSLtoPTXObject.set_stateTexGenCoord("r");}
                            | 'q'       {globalGLSLtoPTXObject.set_stateTexGenCoord("q");}
                        ;

stateClipPlaneItem    : clip arrayMemAbs '.' plane
                        ;

statePointItem        : point '.' statePointProperty
                        ;

statePointProperty    : SIZE                    {globalGLSLtoPTXObject.set_statePointProperty("size");}
                            | attenuation       {globalGLSLtoPTXObject.set_statePointProperty("attenuation");}
                        ;

stateTexEnvItem       : texenv optArrayMemAbs '.' color
                        ;

stateDepthItem        : depth '.' range
                        ;


programSingleItem     : progEnvParam            {globalGLSLtoPTXObject.set_programSingleItem("progEnvParam");}
                            | progLocalParam    {globalGLSLtoPTXObject.set_programSingleItem("progLocalParam");}
                        ;

programMultipleItem   : progEnvParams           {globalGLSLtoPTXObject.set_programMultipleItem("progEnvParams");}
                            | progLocalParams   {globalGLSLtoPTXObject.set_programMultipleItem("progLocalParams");}
                        ;

progEnvParams         : program '.' env arrayMemAbs             {globalGLSLtoPTXObject.set_progEnvParams("arrayMemAbs");}
                            | program '.' env arrayRange        {globalGLSLtoPTXObject.set_progEnvParams("arrayRange");}     
                        ;

progEnvParam          : program '.' env arrayMemAbs
                        ;

progLocalParams       : program '.' local arrayMemAbs           {globalGLSLtoPTXObject.set_progLocalParams("arrayMemAbs");}
                            | program '.' local arrayRange      {globalGLSLtoPTXObject.set_progLocalParams("arrayRange");}
                        ;

progLocalParam        : program '.' local arrayMemAbs
                        ;

constantVector        : '{' constantVectorList '}'      {globalGLSLtoPTXObject.set_constantVector();}
                        ;

constantVectorList    :         signedConstantScalar    {globalGLSLtoPTXObject.add_constantVectorList();}
                            |   signedConstantScalar    {globalGLSLtoPTXObject.add_constantVectorList();}
                            ',' constantVectorList
                        ;

signedConstantScalar  : optSign constantScalar  {globalGLSLtoPTXObject.set_signedConstantScalar();}
                        ;

constantScalar        : floatConstantScalar     {globalGLSLtoPTXObject.set_constantScalar("floatConstantScalar");}
                            | intConstantScalar {globalGLSLtoPTXObject.set_constantScalar("intConstantScalar");}
                        ;

floatConstantScalar   : floatConst      {globalGLSLtoPTXObject.set_floatConstantScalar($1);}
                        ;

intConstantScalar     : intConst        {globalGLSLtoPTXObject.set_intConstantScalar($1);}
                        ;

tempUseS              : tempVarName scalarSuffix
                        ;

tempUseVNS            : tempVarName
                        ;

resultUseD            : resultBasic optFaceType optColorType            {globalGLSLtoPTXObject.set_resultUseD("resultBasic");}
                            | resultMulti                               {globalGLSLtoPTXObject.set_resultUseD("resultMulti");}
                        ;

resultMulti           : resPrefix texcoord arrayRange                   {globalGLSLtoPTXObject.set_resultMulti("texcoord");}
                              | resPrefix clip arrayRange               {globalGLSLtoPTXObject.set_resultMulti("clip");}
                              | resPrefix attrib arrayRange             {globalGLSLtoPTXObject.set_resultMulti("attrib");}
			;

resultBasic           : resPrefix color                                 {globalGLSLtoPTXObject.set_resultBasic("color");}
                              | resPrefix depth                         {globalGLSLtoPTXObject.set_resultBasic("depth");} 
                              | resPrefix position                      {globalGLSLtoPTXObject.set_resultBasic("position");}
                              | resPrefix fogcoord                      {globalGLSLtoPTXObject.set_resultBasic("fogcoord");}
                              | resPrefix pointsize                     {globalGLSLtoPTXObject.set_resultBasic("pointsize");}
                              | resPrefix texcoord optArrayMemAbs       {globalGLSLtoPTXObject.set_resultBasic("texcoord");}
                              | resPrefix clip arrayMemAbs              {globalGLSLtoPTXObject.set_resultBasic("clip");}
                              | resPrefix attrib arrayMemAbs            {globalGLSLtoPTXObject.set_resultBasic("attrib");}
                              | resPrefix ID                            {globalGLSLtoPTXObject.set_resultBasic("ID");}
                              //resultOptColorNum, even it is on the grammer it is useless (see below as resultOptColorNum) is empty and it causes warnings
			;


//resultOptColorNum     : /* empty */	//commented as it causes warnings
//			;

resPrefix             : result '.'
			;

bufferUseS            : bufferVarName optArrayMem scalarSuffix
                        ;

bufferUseVNS          : bufferVarName optArrayMem
                        ;

bufferUseDB           : bufferBinding arrayMemAbs
                        ;

bufferUseDM           : bufferBinding arrayMemAbs
                            | bufferBinding arrayRange
                            | bufferBinding
                        ;

bufferBinding         : program '.' buffer arrayMemAbs
                        ;

optArraySize          : '[' ']'                 {globalGLSLtoPTXObject.set_optArraySize(0);}
                            | '[' intConst ']'  {globalGLSLtoPTXObject.set_optArraySize($2);}
                        ;

optArrayMem           : /* empty */             {globalGLSLtoPTXObject.set_optArrayMem(false);}
                            | arrayMem          {globalGLSLtoPTXObject.set_optArrayMem(true);}
                        ;

arrayMem              : arrayMemAbs             {globalGLSLtoPTXObject.set_arrayMem("arrayMemAbs");}                
                            | arrayMemRel       {cout<<"arrayMemRel is not supported "<<endl; abort();}
                        ;

optArrayMemAbs        : /* empty */             {globalGLSLtoPTXObject.set_optArrayMemAbs(false);}
                            | arrayMemAbs       {globalGLSLtoPTXObject.set_optArrayMemAbs(true);}
                        ;

arrayMemAbs           : '[' intConst ']' {globalGLSLtoPTXObject.set_arrayMemAbs($2);}
                        ;

arrayMemRel           : '[' arrayMemReg arrayMemOffset ']'
                        ;

arrayMemReg           : addrUseS
                        ;

arrayMemOffset        : /* empty */
                            | '+' intConst
                            | '-' intConst
                        ;

optRowArrayRange         : /* empty */                  {globalGLSLtoPTXObject.set_optRowArrayRange(false);}
                           | '.' row arrayRange         {globalGLSLtoPTXObject.set_optRowArrayRange(true);}
                        ;

arrayRange            : '[' intConst TWO_DOTS intConst ']'      {globalGLSLtoPTXObject.set_arrayRange($2,$4);}
                        ;

addrUseS              : addrVarName scalarSuffix
                        ;

optCcMask             :  /*empty */
                           | ccMask    {cout<<"Error: ccMask is not supported\n"; abort();}
                        ;

ccMask                : '(' ccTest ')'
                        ;

ccTest                : ccMaskRule swizzleSuffix
                        ;

ccMaskRule            : EQ
                            | GE
                            | GT
                            | LE
                            | LT
                            | NE
                            | TR
                            | FL
                            | EQ0
                            | GE0
                            | GT0
                            | LE0
                            | LT0
                            | NE0
                            | TR0
                            | FL0
                            | EQ1
                            | GE1
                            | GT1
                            | LE1
                            | LT1
                            | NE1
                            | TR1
                            | FL1
                            | NAN
                            | NAN0
                            | NAN1
                            | LEG
                            | LEG0
                            | LEG1
                            | CF
                            | CF0
                            | CF1
                            | NCF
                            | NCF0
                            | NCF1
                            | OF
                            | OF0
                            | OF1
                            | NOF
                            | NOF0
                            | NOF1
                            | AB
                            | AB0
                            | AB1
                            | BLE
                            | BLE0
                            | BLE1
                            | SF
                            | SF0
                            | SF1
                            | NSF
                            | NSF0
                            | NSF1
                        ;

optWriteMask          : /* empty */                     {globalGLSLtoPTXObject.set_optWriteMask("xyzw");}
                            | '.' xyzwSwizzleMask       {globalGLSLtoPTXObject.set_optWriteMask($2);}
                            | '.' rgbaSwizzleMask       {globalGLSLtoPTXObject.set_optWriteMask($2);}
                        ;

swizzleSuffix         : /* empty */                     {globalGLSLtoPTXObject.set_swizzleSuffix("xyzw");}
                            | '.' xyzwSwizzleMask       {globalGLSLtoPTXObject.set_swizzleSuffix($2);}
                            | '.' rgbaSwizzleMask       {globalGLSLtoPTXObject.set_swizzleSuffix($2);}
                        ;

extendedSwizzle       : extSwizComp ',' extSwizComp ',' 
                            extSwizComp ',' extSwizComp
                        ;

extSwizComp           : optSign xyzwExtSwizSel
                            | optSign rgbaExtSwizSel
                        ;

xyzwExtSwizSel        : '0'
                            | '1'
                            | xyzwSwizzleMask //should check that we have one element here only
                        ;

rgbaExtSwizSel        : rgbaSwizzleMask //should check that we have one element here only
                        ;

scalarSuffix          : '.' component
                        ;

component             : xyzwSwizzleMask //should check that we have one element here only
                            | rgbaSwizzleMask //should check that we have one element here only
                        ;

optSign               : /* empty */     {globalGLSLtoPTXObject.set_optSign(false);}
                            | '-'       {globalGLSLtoPTXObject.set_optSign(true);}
                            | '+'       {globalGLSLtoPTXObject.set_optSign(false);}
                        ;

optFaceType           : /* empty */             {globalGLSLtoPTXObject.set_optFaceType(false);}
                            | '.' faceType      {globalGLSLtoPTXObject.set_optFaceType(true);}
                        ;

faceType              : front           {globalGLSLtoPTXObject.set_faceType("front");}
                            | back      {globalGLSLtoPTXObject.set_faceType("back");}
                        ;

optColorType          : /* empty */             {globalGLSLtoPTXObject.set_optColorType(true);}  
                            | '.' colorType     {globalGLSLtoPTXObject.set_optColorType(false);}

colorType             : primary         {globalGLSLtoPTXObject.set_colorType("primary");}
                            | secondary {globalGLSLtoPTXObject.set_colorType("secondary");}
                        ;

instLabel             : identifier
                        ;

instTarget            : identifier
                        ;

establishedName       : identifier
                        ;

establishName         : identifier
                        ;

tempVarName           : identifier	
                        ;

paramVarName          : identifier	
                        ;

attribVarName         : identifier	
                        ;

resultVarName         : identifier	
                        ;

bufferVarName         : identifier	
                        ;

addrVarName           : identifier	
                        ;

identifier	      : STRING  {globalGLSLtoPTXObject.set_identifier($1);}
			;

%%


int main(int argc, char *argv[]) {
	//yydebug =1;
	assert(argc==10);
	printf("Input ARB file name is %s\n",argv[1]);
	// open a file handle to a particular file:
	FILE *myfile = fopen(argv[1], "r");
        
        bool startIndexOne = true;
        bool zeroIndexIsConstant = false;
        int usedTexUnits = 0;
        char  tmp[256]={0x0};
        while(myfile!=NULL && fgets(tmp, sizeof(tmp),myfile)!=NULL)
        {
            if (strstr(tmp, "c[0]"))
                startIndexOne = false;
            if (strstr(tmp, "#const c[0]"))
                zeroIndexIsConstant = true;
            if (strstr(tmp, "texunit"))
                usedTexUnits++;
        }
        
        rewind(myfile);
        
        globalGLSLtoPTXObject.setPTXFilePath(argv[2], argv[6], startIndexOne or zeroIndexIsConstant,usedTexUnits);
        globalGLSLtoPTXObject.setInShaderBlending(argv[3]);
        globalGLSLtoPTXObject.setBlendingEnabled(argv[4]);
        globalGLSLtoPTXObject.setDepthEnabled(argv[5]);
        globalGLSLtoPTXObject.setShaderFuncName(argv[7]);
        globalGLSLtoPTXObject.setDepthSize(argv[8]);
        globalGLSLtoPTXObject.setDepthFunc(argv[9]);
        
	// make sure it's valid:
	if (!myfile) {
		cout << "I can't open the input ARB file!" << endl;
		exit(1);
	}
	// set lex to read from it instead of defaulting to STDIN:
	yyin = myfile;

	// parse through the input until there is no more:
	
	do {
		yyparse();
	} while (!feof(yyin));
	
        fclose(myfile);
}

void yyerror(string s) {
	cout << "parse error!  Message: " << s << endl;
	// might as well halt now:
	exit(-1);
}
