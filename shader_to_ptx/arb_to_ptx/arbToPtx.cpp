#include <stdlib.h>
#include <string>
#include <iostream>


#include <bits/basic_ios.h>
#include <queue>
#include <map>
#include <vector>
#include <algorithm>
#include <sstream>
#include <assert.h>
#include <iosfwd>
#include <ios>
#include<mtypes.h>
#include <GL/gl.h>
#include <fstream>
#include <bits/ios_base.h>
#include <bits/stl_vector.h>
#include <bits/basic_string.h>
#include <bits/stringfwd.h>
#include <iomanip>
#include <bits/stl_map.h>

#define debugModeValue 0

#define LOG_TRACE(X){\
if(debugMode){\
    std::cout<<"ARB_NV4 to PTX Converter: ";\
    std::cout<<"at "<<__FUNCTION__<<" received "<<"\033[1;32m"<<X<<"\033[0m"<<std::endl;}}

#define HANDLE_TRACE(X){\
if(debugMode){\
    std::cout<<"ARB_NV4 to PTX Converter: ";\
    std::cout<<"at "<<__FUNCTION__<<" handling "<<"\033[1;31m"<<X<<"\033[0m"<<" value"<<std::endl;}}


class Constants{
private:
    std::stringstream ss;
public:
    std::string STR_VECTOR_SIZE;
    std::string STR_PER_VERTEX_DATA_SIZE;
    unsigned int INT_DATA_TYPE_SIZE;
    unsigned int INT_VECTOR_SIZE;
    unsigned int INT_VECTOR_SIZE_IN_BYTES; //vector data size in bytes
    unsigned int INT_PER_VERTEX_RESULT_SIZE;
    std::string STR_PER_VERTEX_RESULT_SIZE;
    unsigned int INT_PER_FRAGMENT_RESULT_SIZE;
    std::string STR_PER_FRAGMENT_RESULT_SIZE;
    Constants(){
        INT_VECTOR_SIZE = 4;
        ss<<INT_VECTOR_SIZE;
        INT_DATA_TYPE_SIZE = sizeof(GLfloat);
        STR_VECTOR_SIZE=ss.str();
        ss.str("");     //clearing the stream
        ss<<VERT_ATTRIB_MAX*INT_VECTOR_SIZE*INT_DATA_TYPE_SIZE;       //in bytes
        STR_PER_VERTEX_DATA_SIZE = ss.str();
        ss.str("");
        INT_VECTOR_SIZE_IN_BYTES = INT_VECTOR_SIZE * INT_DATA_TYPE_SIZE;
        INT_PER_VERTEX_RESULT_SIZE = VERT_RESULT_MAX*INT_VECTOR_SIZE*INT_DATA_TYPE_SIZE;
        ss<<INT_PER_VERTEX_RESULT_SIZE;
        STR_PER_VERTEX_RESULT_SIZE = ss.str();
        ss.str("");
        INT_PER_FRAGMENT_RESULT_SIZE = 2*INT_VECTOR_SIZE*INT_DATA_TYPE_SIZE;
        ss<<INT_PER_FRAGMENT_RESULT_SIZE;
        STR_PER_FRAGMENT_RESULT_SIZE = ss.str();
        ss.str("");
    }
};
 
struct attribBasic_t{
    std::string name;
    std::string type;
};

struct xyzwMask_t{
    bool x;
    bool y;
    bool z;
    bool w;
};

struct arrayRange_t{
    int start;
    int end;
};

struct instructionOperand_t{
    std::string tempAttribParamBufferUseV;
    std::string instOperandBaseV;
    std::string instOperandV;
    std::string attribUseV;
    std::string attribColor;
    bool optColorType;
    std::string colorType;
    attribBasic_t attribBasic;
    std::string type;
    std::string identifier;
    std::string swizzleSuffix;
    bool optArrayMem;
    bool optArrayMemAbs;
    int arrayMemAbs;
    std::string arrayMem;
    
    std::string stateSingleItem;
    std::string stateMatProperty;
    std::string stateLightProperty;
    std::string stateLModProperty;
    std::string stateLProdProperty;
    std::string stateFogProperty;
    bool optStateMatModifier;
    std::string stateMatModifier;
    std::string stateMatrixName;
    std::string stateTexGenType;
    std::string stateTexGenCoord;
    std::string statePointProperty;
    std::string programSingleItem;
    std::vector<float> constantVectorList;
    float signedConstantScalar;
    bool optFaceType;
    std::string faceType;
    bool optSign;
};

struct instructionResult_t{
    std::string identifier;
    std::string optWriteMask;
    std::string type;
    std::string instResultBase;
    bool optFaceType;
    bool optColorType;
    bool optArrayMem;
    bool optArrayMemAbs;
    int arrayMemAbs;
    std::string faceType;
    std::string colorType;
    std::string resultBasic;
};

struct instructionMap_t{
    typedef int mappingType;
    const static mappingType DirectVector2Scalar = 0;
    const static mappingType UnDirectVector2Scalar =1;
    std::string ptxInstruction;
    mappingType type;
};

struct texImageUnit_t{
    int arrayMemAbs;
    bool optArrayMemAbs;
};

struct paramMultInitList_t{
    std::string type;
    std::string paramUseDM;
    std::string paramUseDB;
    std::string stateSingleItem;
    std::vector<float> constantVectorList;
    float signedConstantScalar;
    std::string stateMatProperty;
    std::string stateLightProperty;
    std::string stateLModProperty;
    std::string stateLProdProperty;
    std::string stateFogProperty;
    std::string stateMatModifier;
    std::string stateMatrixName;
    std::string stateTexGenType;
    std::string stateTexGenCoord;
    std::string statePointProperty;
    std::string programSingleItem;
    std::string programMultipleItem;
    std::string progEnvParams;
    std::string progLocalParams;
    arrayRange_t arrayRange;
    int arrayMemAbs;
    bool optArrayMemAbs;
    bool optArrayRange;
    bool optRowArrayRange;
    bool optStateMatModifier;
    bool optFaceType;
    std::string faceType;
};

struct stateData {
    std::string program_type;
    std::string identifier;
    std::string VECTORop;
    std::string SCALARop;
    std::string opModifier;
    std::string ALUInstruction;
    std::string instruction;
    std::string statement;
    std::string BINop;
    std::string TRIop;
    std::string instResult;
    std::string instResultBase;
    std::string optWriteMask;
    int arrayMemAbs;
    std::string instOperandBaseV;
    std::string instOperandV;
    std::string tempAttribParamBufferUseV;
    bool optArrayMem;
    std::string arrayMem;
    std::string swizzleSuffix;
    attribBasic_t attribBasic;
    std::string attribUseV;
    std::string attribGeneric;
    std::string resultBasic;
    bool optArrayMemAbs;
    std::string faceType;
    std::string colorType;
    std::string resultUseW;
    bool optFaceType;
    bool optColorType;
    std::string varModifier;
    bool varMods;
    std::string namingStatement;
    std::string attribMulti;
    std::string attribUseD;
    std::string attribTexCoord;
    std::string attribColor;
    std::vector <instructionOperand_t> operands;
    std::vector <instructionResult_t> destinations;
    std::vector<std::string> varNameList;
    //it is not used but to count the number of modifiers and abort when is more than 1
    std::vector<std::string> opModifierItem;
    arrayRange_t arrayRange;
    std::string resultUseD;
    std::string resultMulti;
    bool optSign;       //if true then the sing is "-", else the sign is "+" or no sign
    std::string stateSingleItem;
    std::string stateMatProperty;
    std::string stateLightProperty;
    std::string stateLModProperty;
    std::string stateLProdProperty;
    std::string stateFogProperty;
    std::string stateTexGenType;
    std::string stateTexGenCoord;
    std::string statePointProperty;
    std::string programSingleItem;
    std::string programMultipleItem;
    std::string progEnvParams;
    std::string progLocalParams;
    float constantScalar;
    float floatConstantScalar;
    int intConstantScalar;
    std::string paramUseDM;
    std::string paramUseDB;
    float signedConstantScalar;
    std::vector<float> constantVectorList;
    bool optArrayRange;
    std::string stateMatrixName;
    std::string stateMatModifier;
    std::vector<paramMultInitList_t> paramMultInitList;
    bool optRowArrayRange;
    bool optStateMatModifier;
    int optArraySize;
    std::string PARAM_statement;
    std::string TexInstruction;
    std::string TEXop;
    std::string texTarget;
    texImageUnit_t texImageUnit;
    bool satFlag;
};

struct dataPassing_t{
    std::string resultValue;
    int requestIndexNumber;
    char maskChar;
    int maskCharIndex;
    int maskCharIntValue;
    int resultWriteMaskIntValue;
};

struct paramMappedVar_t{
    
};

struct mappedVar_t{
    std::string type;
    std::string name;
    bool optArrayMemAbs;
    bool optArrayRange;
    int arrayRange_start;
    int arrayRange_end;
    bool optColorType;
    std::string colorType;
    int arrayMemAbs;
    bool optFaceType;
    std::string faceType;
    std::string mappingType;
    //the next variables are only used by PARAM
    //used to store PARAM values 
    std::vector<paramMultInitList_t> paramMultInitList;
    //used to store PARAM values ranges
    std::vector<arrayRange_t> paramValuesRanges;
};

class arbToPtxConverter{
private:
    typedef void (arbToPtxConverter::*simpleFunctionPointer) ();
    //members
    bool debugMode;
    stateData currentState;
    dataPassing_t passedData;
    std::map<std::string, simpleFunctionPointer> functionCallsTable;
    std::map<std::string, instructionMap_t> instructionMappingTable;
    std::map<std::string, mappedVar_t> variablesMappingTable;
    std::vector<std::string> upperInstructionsList;
    std::vector<std::string> middleInstructionsList;
    std::vector<std::string> lowerInstructionsList;
    std::map<std::string,bool> generatedInstructions;
    std::map<std::string, std::string> modifiersMappingTable;
    std::string vertexFile;
    std::string fragmentFile;
    bool startIndexOne;
    int usedTexUnits;
    Constants constants; 
    bool inShaderBlending;
    bool blendingEnabled;
    bool depthEnabled;
    std::string depthSize;
    std::string depthFunc1;
    std::string depthFunc2;
    std::string fileNo;
    std::string localParamsName;
    std::string shaderFuncName;

    //methods
    void initializeFunctionCallsTable();
    void initializeInstructionMappingTable();
    void initializeModifiersMappingTable();
    void initializeVertexShaderData();
    void initializeFragmentShaderData();
    void loadCommonShaderData();
    std::string getPTXVariable(std::string,int,char,int);
    void clearState();
    void mapOperand(int);
    void mapResult(int);
    unsigned int getMaskIntegerValue(char);
    void addIfDoesNotExist(std::vector<std::string> *, std::string);
    
public:
    arbToPtxConverter(){
        debugMode=debugModeValue;
        clearState();
        initializeFunctionCallsTable();
        initializeModifiersMappingTable();
        initializeInstructionMappingTable();
        inShaderBlending = false;
        blendingEnabled = false;
        depthEnabled = false;
        depthSize = "Z16";
        depthFunc1 = "max";
        depthFunc2 = "gt";
    }

    void setPTXFilePath(char * folder,  char * cfileNo, bool pStartIndexOne, int pUsedTexUnits) {

        std::string fileNo = cfileNo;
        vertexFile = folder;
        vertexFile += "/vertex_shader" + fileNo + ".ptx";
        fragmentFile = folder;
        fragmentFile += "/fragment_shader" + fileNo + ".ptx";
        
        startIndexOne = pStartIndexOne;
        usedTexUnits = pUsedTexUnits;
    }
    
    void setInShaderBlending(std::string isInShaderBlending){
        if(isInShaderBlending=="inShader")
            inShaderBlending = true;
        else if(isInShaderBlending=="disabled")
            inShaderBlending = false;
        else {
            printf("Unrecognized shader blending option\n");
            abort();
        }
    }

    void setBlendingEnabled(std::string value){
       if(value == "enabled"){
          blendingEnabled = true;
       } else {
          blendingEnabled = false;
       }
    }

    void setDepthEnabled(std::string value){
       if(value == "enabled"){
          depthEnabled = true;
       } else {
          depthEnabled = false;
       }
    }

    void setDepthSize(std::string value){
       if(depthEnabled){
          depthSize = value;
       } else {
          depthSize = "DEPTH_DISABLED";
       }
    }


    void setDepthFunc(std::string value){
       int dfun = atoi(value.c_str());
       if(depthEnabled){
          switch(dfun){
             case GL_LESS: 
                depthFunc1 = "min";
                depthFunc2 = "lt";
                break;
             case GL_LEQUAL:
                depthFunc1 = "min";
                depthFunc2 = "le";
                break;
             case GL_GEQUAL: 
                depthFunc1 = "max";
                depthFunc2 = "ge";
                break;
             case GL_GREATER: 
                depthFunc1 = "max";
                depthFunc2 = "gt";
                break;
             default: 
                printf("Unsupported depth function %x\n", dfun);
                abort();
          }
       } else {
          //shouldn't matter
          depthFunc1 = depthFunc2 =  "DEPTH_DISABLED";
       }
    }

    void setShaderFuncName(char * shaderName){
        shaderFuncName = shaderName;
    }
    
    ~arbToPtxConverter(){
    }
    
    //set functions that works around storing the state of each program statement
    //before generating the equivalent PTX statement(s)
    
    void set_VECTORop(std::string pVECTORop, bool pSatFlag){
        LOG_TRACE(pVECTORop)
        currentState.VECTORop = pVECTORop;
        currentState.satFlag = pSatFlag;
    }
    
    void set_SCALARop(std::string pSCALARop){
        LOG_TRACE(pSCALARop)
        currentState.SCALARop = pSCALARop;
    }
    
    void set_BINop(std::string pBINop){
        LOG_TRACE(pBINop)
        currentState.BINop = pBINop;
    }
    
    void set_ALUInstruction(std::string pALUInstruction){
        LOG_TRACE(pALUInstruction)
        currentState.ALUInstruction = pALUInstruction;
    }
    
    void set_TRIop(std::string pTRIop){
        LOG_TRACE(pTRIop)
        currentState.TRIop = pTRIop;
    }
    
    void set_instruction(std::string pInstruction){
        LOG_TRACE(pInstruction)
        currentState.instruction = pInstruction;
    }
    
    void set_statement(std::string pStatement){
        LOG_TRACE(pStatement)
        currentState.statement = pStatement;
    }
    
    void set_opModifier(std::string pOpModifier){
        LOG_TRACE(pOpModifier)
        currentState.opModifier = pOpModifier;
    }
    
    void set_instResult(std::string pInstResult){
        LOG_TRACE(pInstResult)
        currentState.instResult = pInstResult;
    }
    
    void set_instResultBase(std::string pInstResultBase){
        LOG_TRACE(pInstResultBase)
        currentState.instResultBase = pInstResultBase;
    }
    
    void set_identifier(std::string pIdentifier){
        LOG_TRACE(pIdentifier)
        currentState.identifier = pIdentifier;
    }
    
    void set_optWriteMask(std::string pOptWriteMask){
        LOG_TRACE(pOptWriteMask)
        currentState.optWriteMask = pOptWriteMask;
    }
    
    void set_arrayMemAbs(int pArrayMemAbs){
        LOG_TRACE(pArrayMemAbs)
        currentState.arrayMemAbs = pArrayMemAbs;       
    }
    
    void set_instOperandBaseV(std::string pInstOperandBaseV){
        LOG_TRACE(pInstOperandBaseV)
        currentState.instOperandBaseV = pInstOperandBaseV;
    }
    
    void set_instOperandV(std::string pInstOperandV){
        LOG_TRACE(pInstOperandV)
        currentState.instOperandV = pInstOperandV;
    }
    
    void set_tempAttribParamBufferUseV(std::string ptempAttribParamBufferUseV){
        LOG_TRACE(ptempAttribParamBufferUseV)
        currentState.tempAttribParamBufferUseV = ptempAttribParamBufferUseV;
    }
    
    void set_optArrayMem(bool pOptArrayMem){
        LOG_TRACE(pOptArrayMem)
        currentState.optArrayMem = pOptArrayMem;
    }
    
    void set_arrayMem(std::string pArrayMem){
        LOG_TRACE(pArrayMem)
        currentState.arrayMem = pArrayMem;
    }
    
    void set_swizzleSuffix(std::string pSwizzleSuffix){
        LOG_TRACE(pSwizzleSuffix)
        std::replace(pSwizzleSuffix.begin(),pSwizzleSuffix.end(),'r','x');
        std::replace(pSwizzleSuffix.begin(),pSwizzleSuffix.end(),'g','y');
        std::replace(pSwizzleSuffix.begin(),pSwizzleSuffix.end(),'b','z');
        std::replace(pSwizzleSuffix.begin(),pSwizzleSuffix.end(),'a','w');
        assert(pSwizzleSuffix.size()!=2 && pSwizzleSuffix.size()!=3);
        if(pSwizzleSuffix.size()==1) currentState.swizzleSuffix = pSwizzleSuffix+pSwizzleSuffix+pSwizzleSuffix+pSwizzleSuffix;
        else currentState.swizzleSuffix = pSwizzleSuffix;
    }
    
    void set_attribBasic(std::string pAttribBasic_Type, std::string pAttribBasic){
        LOG_TRACE(pAttribBasic)
        currentState.attribBasic.name = pAttribBasic;
        currentState.attribBasic.type = pAttribBasic_Type;
        if(currentState.attribBasic.type=="attribTexCoord"){
            currentState.attribBasic.type = currentState.attribTexCoord;
            currentState.attribBasic.name = "texcoord";
        } else if(currentState.attribBasic.type=="attribGeneric"){
            currentState.attribBasic.type = currentState.attribGeneric;
            currentState.attribBasic.name = "attrib";
        } else if(currentState.attribBasic.type=="attribClip"){
            currentState.attribBasic.type = "fragment";
            currentState.attribBasic.name = "clip";
        }
    }
    
    void set_arrayRange(int pStart, int pEnd){
        std::stringstream ss;
        ss<<"Start = "<<pStart<<", End = "<<pEnd;
        LOG_TRACE(ss.str())
        currentState.arrayRange.start = pStart;
        currentState.arrayRange.end = pEnd;
    }
    
    void set_attribUseV(std::string pAttribUseV){
        LOG_TRACE(pAttribUseV)
        currentState.attribUseV = pAttribUseV;
    }
    
    void set_attribTexCoord(std::string pAttribTexCoord){
        LOG_TRACE(pAttribTexCoord)
        currentState.attribTexCoord = pAttribTexCoord;
    }
    
    void set_attribGeneric(std::string pAttribGeneric){
        LOG_TRACE(pAttribGeneric)
        currentState.attribGeneric = pAttribGeneric;
    }
    
    void set_resultBasic(std::string pResultBasic){
        LOG_TRACE(pResultBasic)
        currentState.resultBasic = pResultBasic;
    }
    
    void set_optArrayMemAbs(bool pOptArrayMemAbs){
        LOG_TRACE(pOptArrayMemAbs)
        currentState.optArrayMemAbs = pOptArrayMemAbs;  
    }
    
    void set_faceType(std::string pFaceType){
        LOG_TRACE(pFaceType)
        currentState.faceType = pFaceType;
    }
    
    void set_colorType(std::string pColorType){
        LOG_TRACE(pColorType)
        currentState.colorType = pColorType;
    }
    
    void set_resultUseW(std::string pResultUseW){
        LOG_TRACE(pResultUseW)
        currentState.resultUseW = pResultUseW;
    }
    
    void set_optFaceType(bool pOptFaceType){
        LOG_TRACE(pOptFaceType)
        currentState.optFaceType = pOptFaceType;
    }
    
    void set_optColorType(bool pOptColorType){
        LOG_TRACE(pOptColorType)
        currentState.optColorType;
    }
    
    void set_opModifierItem(){
        currentState.opModifierItem.push_back(currentState.opModifier);
        if(currentState.opModifierItem.size()>1){
            std::cout<<"Multi opModifier are not supported currently"<<std::endl;
            abort();
        }
        LOG_TRACE("Pushing into the queue of modifiers "+currentState.opModifier)
    }
    
    void set_namingStatement(std::string pNamingStatement){
        LOG_TRACE(pNamingStatement)
        currentState.namingStatement = pNamingStatement;
    }
    
    void set_varNameList(){
        LOG_TRACE(currentState.identifier)
        currentState.varNameList.push_back(currentState.identifier);
    }
    
    void set_varMods(bool pVarMods){
        LOG_TRACE(pVarMods)
        currentState.varMods = pVarMods;
        if(debugMode)
            std::cout<<"Warning: Only one varModifier is supported\n";
    }
    
    void set_varModifier(std::string pVarModifier){
        LOG_TRACE(pVarModifier)
        currentState.varModifier = pVarModifier;
    }
    
    void set_attribMulti(std::string pAttribMulti){
        LOG_TRACE(pAttribMulti)
        currentState.attribMulti = pAttribMulti;
    }
    
    void set_attribColor(std::string pAttribColor){
        LOG_TRACE(pAttribColor)
        currentState.attribColor = pAttribColor;
    }
    
    void set_attribUseD(std::string pAttribUseD){
        LOG_TRACE(pAttribUseD)
        currentState.attribUseD = pAttribUseD;
    }
    
    void set_resultUseD(std::string pResultUseD){
        LOG_TRACE(pResultUseD)
        currentState.resultUseD = pResultUseD;
    }
    
    void set_resultMulti(std::string pResultMulti){
        LOG_TRACE(pResultMulti)
        currentState.resultMulti = pResultMulti;
    }
    
    void set_program_type(std::string pProgram_type){
        LOG_TRACE(pProgram_type)
        currentState.program_type = pProgram_type;
        if(pProgram_type=="VERTEX_PROGRAM")
            initializeVertexShaderData();
        else if(pProgram_type=="FRAGMENT_PROGRAM")
            initializeFragmentShaderData();
        else abort();
    }
    
    void set_optSign(bool pOptSign){
        LOG_TRACE(pOptSign)
        currentState.optSign = pOptSign;
    }
    
    void set_stateSingleItem(std::string pStateSingleItem){
        LOG_TRACE(pStateSingleItem)
        currentState.stateSingleItem = pStateSingleItem;
    }
    
    void set_stateMatProperty(std::string pStateMatProperty){
        LOG_TRACE(pStateMatProperty)
        currentState.stateMatProperty = pStateMatProperty;
    }
    
    void set_stateLightProperty(std::string pStateLightProperty){
        LOG_TRACE(pStateLightProperty)
        currentState.stateLightProperty = pStateLightProperty;
    }
    
    void set_stateLModProperty(std::string pStateLModProperty){
        LOG_TRACE(pStateLModProperty)
        currentState.stateLModProperty = pStateLModProperty;
    }
    
    void set_stateLProdProperty(std::string pStateLProdProperty){
        LOG_TRACE(pStateLProdProperty)
        currentState.stateLProdProperty = pStateLProdProperty;
    }
    
    void set_stateFogProperty(std::string pStateFogProperty){
        LOG_TRACE(pStateFogProperty)
        currentState.stateFogProperty = pStateFogProperty;
    }
    
    void set_stateTexGenType(std::string pStateTexGenType){
        LOG_TRACE(pStateTexGenType)
        currentState.stateTexGenType = pStateTexGenType;
    }
    
    void set_stateTexGenCoord(std::string pStateTexGenCoord){
        LOG_TRACE(pStateTexGenCoord)
        currentState.stateTexGenCoord = pStateTexGenCoord;
    }
    
    void set_statePointProperty(std::string pStatePointProperty){
        LOG_TRACE(pStatePointProperty)
        currentState.statePointProperty = pStatePointProperty;
    }
    
    void set_programSingleItem(std::string pProgramSingleItem){
        LOG_TRACE(pProgramSingleItem)
        currentState.programSingleItem = pProgramSingleItem;
    }
    
    void set_programMultipleItem(std::string pProgramMultipleItem){
        LOG_TRACE(pProgramMultipleItem)
        currentState.programMultipleItem = pProgramMultipleItem;
    }
    
    void set_progEnvParams(std::string pProgEnvParams){
        LOG_TRACE(pProgEnvParams)
        currentState.progEnvParams = pProgEnvParams;
    }
    
    void set_progLocalParams(std::string pProgLocalParams){
        LOG_TRACE(pProgLocalParams)
        currentState.progLocalParams = pProgLocalParams;
    }
    
    void set_constantScalar(std::string pConstantScalar){
        LOG_TRACE(pConstantScalar)
        if(pConstantScalar=="floatConstantScalar")
            currentState.constantScalar = currentState.floatConstantScalar;
        else if (pConstantScalar=="intConstantScalar")
            currentState.constantScalar = currentState.intConstantScalar;
        else abort();
    }
    
    void set_floatConstantScalar(float pFloatConstantScalar){
        LOG_TRACE(pFloatConstantScalar)
        currentState.floatConstantScalar = pFloatConstantScalar;
    }
    
    void set_intConstantScalar(int pIntConstantScalar){
        LOG_TRACE(pIntConstantScalar)
        currentState.intConstantScalar = pIntConstantScalar;
    }
    
    void set_paramUseDM(std::string pParamUseDM){
        LOG_TRACE(pParamUseDM)
        currentState.paramUseDM = pParamUseDM;
    }
    
    void set_paramUseDB(std::string pParamUseDB){
        LOG_TRACE(pParamUseDB)
        currentState.paramUseDB = pParamUseDB;
    }
    
    void set_constantVector(){
        std::stringstream ss;
        ss<<"with vector size of "<<currentState.constantVectorList.size();
        LOG_TRACE(ss.str());
        if(currentState.constantVectorList.size()>4){
            std::cout<<"Constant vector size should not exceed 4"<<std::endl;
            abort();
        }
    }
    
    void set_signedConstantScalar(){
        LOG_TRACE(currentState.constantScalar)
        if(currentState.optSign)
            currentState.signedConstantScalar = currentState.constantScalar* -1;
        else currentState.signedConstantScalar = currentState.constantScalar;
    }
    
    void add_constantVectorList(){
        LOG_TRACE(currentState.signedConstantScalar)
        currentState.constantVectorList.push_back(currentState.signedConstantScalar);
    }
    
    void set_stateMatrixName(std::string pStateMatrixName){
        LOG_TRACE(pStateMatrixName)
        currentState.stateMatrixName = pStateMatrixName;
    }
    
    void set_stateMatModifier(std::string pStateMatModifier){
        LOG_TRACE(pStateMatModifier)
        currentState.stateMatModifier = pStateMatModifier;
    }
    
    void set_PARAM_statement(std::string pPARAM_statement){
        LOG_TRACE(pPARAM_statement)
        currentState.PARAM_statement = pPARAM_statement;
    }
    
    void set_TexInstruction(std::string pTexInstruction){
        LOG_TRACE(pTexInstruction)
        currentState.TexInstruction = pTexInstruction;
    }
    
    void set_TEXop(std::string pTEXop){
        LOG_TRACE(pTEXop)
        currentState.TEXop = pTEXop;
    }
    
    void set_texTarget(std::string pTexTarget){
        LOG_TRACE(pTexTarget)
        currentState.texTarget = pTexTarget;
    }
    
    void set_texImageUnit(){
        LOG_TRACE("texture")
        currentState.texImageUnit.optArrayMemAbs = currentState.optArrayMemAbs;
        currentState.texImageUnit.arrayMemAbs = currentState.arrayMemAbs;
    }
    
    void add_paramMultInitList(std::string pType){
        LOG_TRACE(" item")
        paramMultInitList_t item;
        item.type =                     pType;
        item.arrayMemAbs =              currentState.arrayMemAbs;
        item.arrayRange =               currentState.arrayRange;
        item.constantVectorList =       currentState.constantVectorList;
        item.faceType =                 currentState.faceType;
        item.optArrayMemAbs =           currentState.optArrayMemAbs;
        item.optArrayRange =            currentState.optArrayRange;
        item.optFaceType =              currentState.optFaceType;
        item.optRowArrayRange =         currentState.optRowArrayRange;
        item.optStateMatModifier =      currentState.optStateMatModifier;
        item.paramUseDM =               currentState.paramUseDM;
        item.progEnvParams =            currentState.progEnvParams;
        item.progLocalParams =          currentState.progLocalParams;
        item.programMultipleItem =      currentState.programMultipleItem;
        item.programSingleItem =        currentState.programSingleItem;
        item.signedConstantScalar=      currentState.signedConstantScalar;
        item.stateFogProperty=          currentState.stateFogProperty;
        item.stateLModProperty=         currentState.stateLModProperty;
        item.stateLProdProperty=        currentState.stateLProdProperty;
        item.stateLightProperty=        currentState.stateLightProperty;
        item.stateMatModifier=          currentState.stateMatModifier;
        item.stateMatProperty=          currentState.stateMatProperty;
        item.stateMatrixName=           currentState.stateMatrixName;
        item.statePointProperty=        currentState.statePointProperty;
        item.stateSingleItem=           currentState.stateSingleItem;
        item.stateTexGenCoord=          currentState.stateTexGenCoord;
        item.stateTexGenType=           currentState.stateTexGenType;
        item.paramUseDB=                currentState.paramUseDB;
        currentState.constantVectorList.clear(); //so PARAM can have multi vector constants
        currentState.paramMultInitList.push_back(item);
    }
    
    void set_optRowArrayRange(bool pOptRowArrayRange){
        LOG_TRACE(pOptRowArrayRange)
        currentState.optRowArrayRange = pOptRowArrayRange;
    }
    
    void set_optStateMatModifier(bool pOptStateMatModifier){
        LOG_TRACE(pOptStateMatModifier)
        currentState.optStateMatModifier = pOptStateMatModifier;
    }
    
    void set_optArraySize(int pOptArraySize){
        LOG_TRACE(pOptArraySize)
        currentState.optArraySize= pOptArraySize;
    }
    //--------------------------------------------
    
    void set_statementFound(){
        (this->*functionCallsTable["statement"])();
        LOG_TRACE("\033[1;36mEND OF STATEMENT\033[0m")
        clearState();
    }
    std::string replaceSubstring(std::string source, std::string oldSub,std::string newSub){
            size_t found = source.find( oldSub );
            while (found!=std::string::npos){
                source.replace( found, oldSub.size() , newSub );
                found = source.find(oldSub);
            };
            return source;
    }
    
    void placeFragmentWrites(){
        std::string dsize = depthSize == "Z16"? "u16" : "u32";
        for(int i=0;i<lowerInstructionsList.size();i++){
            if(lowerInstructionsList[i].find("ColorWrite0;")==std::string::npos) continue;
            std::stringstream ss;
            ss<<".reg .u32 %r, %b, %g, %a, %bgra;\n";
            ss<<"mul.sat.f32 tempReg0, result_color0_x, 0f3f800000; //Clamp R to 0-1\n";
            ss<<"mul.sat.f32 tempReg1, result_color0_y, 0f3f800000; //Clamp G to 0-1\n";
            ss<<"mul.sat.f32 tempReg2, result_color0_z, 0f3f800000; //Clamp B to 0-1\n";
            ss<<"mul.sat.f32 tempReg3, result_color0_w, 0f3f800000; //Clamp A to 0-1\n";
            ss<<"mul.f32 tempReg0, tempReg0, 0f437f0000; //R*255\n";
            ss<<"mul.f32 tempReg1, tempReg1, 0f437f0000; //G*255\n";
            ss<<"mul.f32 tempReg2, tempReg2, 0f437f0000; //B*255\n";
            ss<<"mul.f32 tempReg3, tempReg3, 0f437f0000; //A*255\n";
            ss<<"cvt.rni.u32.f32 %r, tempReg0;\n";
            ss<<"cvt.rni.u32.f32 %g, tempReg1;\n";
            ss<<"cvt.rni.u32.f32 %b, tempReg2;\n";
            ss<<"cvt.rni.u32.f32 %a, tempReg3;\n";
            
            //ss<<"shl.b32 %b, %b, 0;\n";
            ss<<"shl.b32 %g, %g, 8;\n";            
            ss<<"shl.b32 %r, %r, 16;\n";
            ss<<"shl.b32 %a, %a, 24;\n";
            
            ss<<"or.b32 %bgra, %b, %g;\n";
            ss<<"or.b32 %bgra, %bgra, %r;\n";
            ss<<"or.b32 %bgra, %bgra, %a;\n";
           
            if(depthEnabled){
               ss<<".reg ." << dsize << " %oldDepth, %depth;\n";
               ss<<".reg .u32" << "%tmpCol1, %tmpCol2;";
               ss<<"mul.f32 tempReg0, %fragment_position.z, 0f477fff00; //depth * (2^16)-1\n";
               ss<<"cvt.rni." << dsize << ".f32 %depth, tempReg0;\n";
               ss<<".reg .pred wr, wr2;\n";
               ss<<"depthTest:\n";
               ss<<"ld.global.u32 %tmpCol1, [p_result_data];\n";
               ss<<"atom.global."<< depthFunc1 <<"." << dsize << " %oldDepth, [p_depth_data], %depth;\n";
               ss<<"setp." << depthFunc2 << "." << dsize << " wr, %depth, %oldDepth;\n";

               ss<<"@!wr bra done;\n";
               ss<<"atom.global.cas.u32  %tmpCol2,[p_result_data], %tmpCol1,%bgra;\n";
               ss<<"setp.eq.u32 wr2, %tmpCol1, %tmpCol2;\n";
               ss<<"@wr2 bra done;\n";
               ss<<"bra depthTest;\n";

               if(blendingEnabled){
                  if(inShaderBlending){
                      ss<<".reg .u32 %framebuffer;\n";
                      ss<<"@wr ld.global.u32 %framebuffer, [p_result_data];\n";
                      ss<<"@wr blend.u32 %bgra,%bgra,%framebuffer;\n";
                  }
               }
               ss<<"@wr st.global.u32 [p_result_data], %bgra;\n";
            } else {
               if(blendingEnabled){
                  if(inShaderBlending){
                      ss<<".reg .u32 %framebuffer;\n";
                      ss<<"ld.global.u32 %framebuffer, [p_result_data];\n";
                      ss<<"blend.u32 %bgra,%bgra,%framebuffer;\n";
                  }
               }
               ss<<"st.global.u32 [p_result_data], %bgra;";
            }
            ss<<"done:\n"; 
            lowerInstructionsList[i]=ss.str();
        }
    }
    
    void set_endOfProgram(){
        std::ostream *output;
        placeFragmentWrites();
        
        std::ofstream resultFile;
        
        if(currentState.program_type=="VERTEX_PROGRAM")
            resultFile.open(vertexFile.c_str());
        else if(currentState.program_type=="FRAGMENT_PROGRAM")
            resultFile.open(fragmentFile.c_str());
        else abort();
        
        output = &resultFile;
        
        lowerInstructionsList.push_back("exit;");
        lowerInstructionsList.push_back("}");
        //(this->*functionCallsTable["test"])();
        for(int i=0;i<upperInstructionsList.size();i++)
            *output<<upperInstructionsList[i]<<std::endl;
        for(int i=0;i<middleInstructionsList.size();i++)
            *output<<middleInstructionsList[i]<<std::endl;
        for(int i=0;i<lowerInstructionsList.size();i++)
            *output<<lowerInstructionsList[i]<<std::endl;  
        
        resultFile.close();
    }
     
    //handler functions that generate the ptx code
    void handler_statement(){
        HANDLE_TRACE(currentState.statement)
        (this->*functionCallsTable[currentState.statement])();
    }
    
    void handler_instruction(){
        HANDLE_TRACE(currentState.instruction)
        (this->*functionCallsTable[currentState.instruction])();
    }
    
    void handler_ALUInstruction(){
        HANDLE_TRACE(currentState.ALUInstruction)
        (this->*functionCallsTable[currentState.ALUInstruction])();
    }
    
    void handler_TexInstruction(){
        HANDLE_TRACE(currentState.TexInstruction)
        (this->*functionCallsTable[currentState.TexInstruction])();
    }
    
    void handler_TEXop_instruction(){
        HANDLE_TRACE(currentState.TEXop)
        generateInstruction(currentState.TEXop,0);
    }
    
    void handler_VECTORop_instruction(){
        HANDLE_TRACE(currentState.VECTORop)
        generateInstruction(currentState.VECTORop,1);
    }
    
    void handler_SCALARop_instruction(){
        HANDLE_TRACE(currentState.SCALARop)
        generateInstruction(currentState.SCALARop,1);
    }
    
    void handler_BINop_instruction(){
        HANDLE_TRACE(currentState.BINop)      
        generateInstruction(currentState.BINop,2);
    }
    
    void handler_TRIop_instruction(){
        HANDLE_TRACE(currentState.TRIop)      
        generateInstruction(currentState.TRIop,3);
    }
    
    void generateInstruction(std::string pInstructionType, int pNumberOfOperands){
        HANDLE_TRACE("Generating instruction")
        instructionMap_t currentInstruction = instructionMappingTable[pInstructionType];
        if(currentState.satFlag) 
            currentInstruction.ptxInstruction = currentInstruction.ptxInstruction + ".sat";
        //default value
        std::string ptxModifier = modifiersMappingTable["F"];
        //this should be modified if more than one modifier is used
        if(!currentState.opModifierItem.empty()) 
            ptxModifier = modifiersMappingTable[currentState.opModifierItem[0]];
        
        //if the target is same as destination "vector" register we use temp to prevent the register to write itself in the middle 
        //of operations
        bool useTemp = false;
        std::string resultRegister =getPTXVariable("instResult",0,'x',0);
        resultRegister = resultRegister.substr(0,resultRegister.size()-1);
        std::vector<std::string> operandRegisters;
        for(int j=0; j<pNumberOfOperands; j++){
            std::string operandRegister=getPTXVariable("instOperand",j,'x',0);
            operandRegister = operandRegister.substr(0,operandRegister.size()-1);
            operandRegisters.push_back(operandRegister);
        }
        
        for(int k=0;k<operandRegisters.size();k++)
            if(operandRegisters[k]==resultRegister) useTemp=true;
        
        
        std::map<int,std::string> resultDestinations;
        if(useTemp)
            for(int i=0;i<currentState.optWriteMask.size();i++){
            std::stringstream tempStream;
            tempStream<<"tempReg"<<i;
            resultDestinations[i]=tempStream.str();
            }
        else 
            for(int i=0;i<currentState.optWriteMask.size();i++)
                resultDestinations[i]=getPTXVariable("instResult",0,currentState.optWriteMask[i],i);
            
        
        if(currentInstruction.type==currentInstruction.DirectVector2Scalar){
            for(int i=0;i<currentState.optWriteMask.size();i++){
                std::stringstream instruction;
                instruction<<currentInstruction.ptxInstruction<<ptxModifier<<" "<<resultDestinations[i];
                for(int j=0; j<pNumberOfOperands; j++)
                    instruction<<", "<<getPTXVariable("instOperand",j,currentState.optWriteMask[i], i);
                instruction<<";";
                middleInstructionsList.push_back(instruction.str());
            }
        } else if(currentInstruction.type==currentInstruction.UnDirectVector2Scalar){
            if(pInstructionType=="DP3" || pInstructionType=="DP4"){
                int size;
                size = pInstructionType=="DP3"? size =3 : size =4;
                for(int i=0;i<size;i++){
                    //assert(currentState.optWriteMask.size()==1);
                    std::stringstream instruction;
                    std::string fakeMask = "xyzw";
                    if(i==0) instruction<<"mul"; else instruction<<"mad";
                    instruction<<ptxModifier<<" "<<resultDestinations[0];
                    
                    if(i==0){
                    for(int j=0; j<pNumberOfOperands; j++)
                        instruction<<", "<<getPTXVariable("instOperand",j,fakeMask[i], i);  
                    }
                    else{
                        for(int j=0; j<pNumberOfOperands; j++)
                            instruction<<", "<<getPTXVariable("instOperand",j,fakeMask[i], i);
                        instruction<<", "<<resultDestinations[0];
                    }
                    instruction<<";";
                    middleInstructionsList.push_back(instruction.str());
                }
                
                std::stringstream copyInstruction;
                for(int i=0;i<currentState.optWriteMask.size();i++){
                    if(i==0 && !useTemp) continue; //if temp are not used skip the first copy is redundant
                    copyInstruction.str("");
                    copyInstruction<<"mov"<<ptxModifier<<" "<<getPTXVariable("instResult",0,currentState.optWriteMask[i],i)<<", "<<resultDestinations[0]<<";";
                    middleInstructionsList.push_back(copyInstruction.str());
                }
                useTemp = false; //copying is already done here
            } else if(pInstructionType=="CMP"){
                std::string predicateDef = ".reg .pred tempPred;";
                addIfDoesNotExist(&upperInstructionsList,predicateDef);
                std::stringstream instruction;
                for(int i=0;i<currentState.optWriteMask.size();i++){
                    if(i==0 or currentState.operands[0].swizzleSuffix[i]!=currentState.operands[0].swizzleSuffix[i-1]){
                        instruction<<"setp.lt"<<ptxModifier<<" "<<"tempPred, "
                                <<getPTXVariable("instOperand",0,currentState.optWriteMask[i],i)<<", "
                                <<"0f00000000;";
                        middleInstructionsList.push_back(instruction.str());
                    }
                instruction.str("");
                instruction<<"selp"<<ptxModifier<<" "<<resultDestinations[i]<<", "
                        <<getPTXVariable("instOperand",1,currentState.optWriteMask[i],i)<<", "
                        <<getPTXVariable("instOperand",2,currentState.optWriteMask[i],i)<<", "
                        <<"tempPred;";
                middleInstructionsList.push_back(instruction.str());
                instruction.str("");
                }
            }
            else if(pInstructionType=="TEX"){
                std::stringstream instruction;
                instruction<<"tex."<<currentState.texTarget<<".v4.u8.f32 { ";
                assert(currentState.texTarget=="2d");
                unsigned textureUnitNumber = 0;
                if(currentState.optArrayMemAbs) textureUnitNumber = currentState.arrayMemAbs;
                std::stringstream textureUnitName;
                textureUnitName<<"textureReference"<<(usedTexUnits-textureUnitNumber-1)<<"_2D";
                
                instruction<<"tempReg0, tempReg1, tempReg2, tempReg3}, [";
//                instruction<<getPTXVariable("instResult",0,'x',0)<<", "
//                           <<getPTXVariable("instResult",0,'y',1)<<", "
//                           <<getPTXVariable("instResult",0,'z',2)<<", "
//                           <<getPTXVariable("instResult",0,'w',3)<<"}, [";
                instruction<<textureUnitName.str()<<", {";
                instruction<<"tempReg0, tempReg1}];";
                addIfDoesNotExist(&upperInstructionsList,".tex .u64 "+textureUnitName.str()+";");
                middleInstructionsList.push_back("\
mov.f32 tempReg0, "+getPTXVariable("instOperand",0,'x',0)+";\n\
mov.f32 tempReg1, "+getPTXVariable("instOperand",0,'y',1)+";");
                middleInstructionsList.push_back(instruction.str());
                
                
                for(int i=0; i<currentState.optWriteMask.size();i++){
                    std::stringstream texResultWrite;
                    texResultWrite<<"mov.f32 "<<getPTXVariable("instResult",0,currentState.optWriteMask[i],i)
                            <<", tempReg"<<getMaskIntegerValue(currentState.optWriteMask[i])<<";";
                    middleInstructionsList.push_back(texResultWrite.str());
                }
                useTemp = false;//disabled not the case here
            }
            else abort();
        } else abort();
        
        if(useTemp){
            for(int i=0;i<currentState.optWriteMask.size();i++){
                std::stringstream instruction;
                instruction<<"mov"<<ptxModifier<<" "<<getPTXVariable("instResult",0,currentState.optWriteMask[i],i)<<", tempReg"<<i<<";";
                middleInstructionsList.push_back(instruction.str());
            }
        }
    }
    
    void handler_instResult(){
        HANDLE_TRACE(currentState.destinations[passedData.requestIndexNumber].type)
        (this->*functionCallsTable[currentState.destinations[passedData.requestIndexNumber].type])();
    }
    
    void handler_instResultBase(){
        HANDLE_TRACE(currentState.destinations[passedData.requestIndexNumber].instResultBase)
        (this->*functionCallsTable[currentState.destinations[passedData.requestIndexNumber].instResultBase])();
    }
      
    void handler_resultBasic(){
        HANDLE_TRACE(currentState.destinations[passedData.requestIndexNumber].instResultBase)
         //zzzz TO DO need to be rechecked
        bool fragmentColorWrite = false;
        instructionResult_t* result = &currentState.destinations[passedData.requestIndexNumber];
        std::stringstream varName;
        varName<<"result_";
        std::stringstream resultRegName;
        assert(!result->optColorType && ! result->optFaceType); //just for testing
        if(currentState.program_type=="VERTEX_PROGRAM")
            resultRegName<<"%vertex_";
        else if(currentState.program_type=="FRAGMENT_PROGRAM")
            resultRegName<<"%fragment_";
        else abort();
        
        
        if(result->resultBasic=="position"){
            varName<<"position_"<<passedData.maskChar;
            resultRegName<<"position."<<passedData.maskChar;
        }
        else if(result->resultBasic=="color"){
            if(result->optColorType && result->colorType=="secondary"){
                varName<<"color1_"<<passedData.maskChar;
                resultRegName<<"color1."<<passedData.maskChar;
            } else {
                varName<<"color0_"<<passedData.maskChar;
                resultRegName<<"color0."<<passedData.maskChar;
            }
            
            if(currentState.program_type=="FRAGMENT_PROGRAM")
                fragmentColorWrite = true;
        }
        else if(result->resultBasic=="fogcoord"){
            varName<<"fogcoord_"<<passedData.maskChar;
            resultRegName<<"fogcoord."<<passedData.maskChar;
        }
        else if(result->resultBasic=="texcoord"){
            int arrayIndex =0;
            if(result->optArrayMemAbs) arrayIndex= result->arrayMemAbs;
            varName<<"texcoord"<<arrayIndex<<passedData.maskChar;
            resultRegName<<"texcoord"<<arrayIndex<<"."<<passedData.maskChar;
        }
        else if(result->resultBasic=="normal"){
            varName<<"normal_"<<passedData.maskChar;
            resultRegName<<"normal."<<passedData.maskChar;
        }
        std::string relatedDiffStatement = ".reg .f32 "+varName.str()+";";
        addIfDoesNotExist(&upperInstructionsList,relatedDiffStatement);
        
        std::stringstream correspondentStoreInstruction;
        if(!fragmentColorWrite) correspondentStoreInstruction<<"mov.f32 "<<resultRegName.str()<<", " <<varName.str()<<";";
        else correspondentStoreInstruction<<"ColorWrite"<<(result->optColorType and result->colorType=="secondary")<<";";
        
        addIfDoesNotExist(&lowerInstructionsList,correspondentStoreInstruction.str());
        passedData.resultValue = varName.str();
    }
    
    void handler_resultVarName(){
        HANDLE_TRACE(currentState.destinations[passedData.requestIndexNumber].identifier)
        std::stringstream ss;
        ss<<currentState.destinations[passedData.requestIndexNumber].identifier;
        if(currentState.destinations[passedData.requestIndexNumber].optArrayMem){
            ss<<currentState.destinations[passedData.requestIndexNumber].arrayMemAbs;
        }
        ss<<passedData.maskChar;
        passedData.resultValue= ss.str();
         //now this one is ready zzzz
    }
    
    void add_instOperand(std::string pType){
        LOG_TRACE("Adding " + pType)
        //(this->*functionCallsTable["instOperandV"])();
        instructionOperand_t instructionOperand;
        instructionOperand.type = pType;
        instructionOperand.swizzleSuffix = currentState.swizzleSuffix;
        instructionOperand.instOperandV = currentState.instOperandV;
        instructionOperand.instOperandBaseV = currentState.instOperandBaseV;
        instructionOperand.optArrayMem = currentState.optArrayMem;
        //arrayMemRel not supported so just store arrayMemAbs
        instructionOperand.arrayMemAbs = currentState.arrayMemAbs;
        instructionOperand.tempAttribParamBufferUseV = currentState.tempAttribParamBufferUseV;
        instructionOperand.identifier = currentState.identifier;
        instructionOperand.attribUseV = currentState.attribUseV;
        instructionOperand.attribColor = currentState.attribColor;
        instructionOperand.attribBasic.name = currentState.attribBasic.name;
        instructionOperand.attribBasic.type = currentState.attribBasic.type;
        instructionOperand.optColorType = currentState.optColorType;
        instructionOperand.colorType = currentState.colorType;
        instructionOperand.arrayMem = currentState.arrayMem;
        instructionOperand.optArrayMemAbs = currentState.optArrayMemAbs;
        instructionOperand.stateSingleItem = currentState.stateSingleItem;
        instructionOperand.stateMatProperty = currentState.stateMatProperty;
        instructionOperand.stateLightProperty = currentState.stateLightProperty;
        instructionOperand.stateLModProperty = currentState.stateLModProperty;
        instructionOperand.stateLProdProperty = currentState.stateLProdProperty;
        instructionOperand.stateFogProperty = currentState.stateFogProperty;
        instructionOperand.optStateMatModifier = currentState.optStateMatModifier;
        instructionOperand.stateMatModifier = currentState.stateMatModifier;
        instructionOperand.stateMatrixName = currentState.stateMatrixName;
        instructionOperand.stateTexGenType = currentState.stateTexGenType;
        instructionOperand.stateTexGenCoord = currentState.stateTexGenCoord;
        instructionOperand.statePointProperty = currentState.statePointProperty;
        instructionOperand.programSingleItem = currentState.programSingleItem;
        instructionOperand.constantVectorList = currentState.constantVectorList;
        instructionOperand.signedConstantScalar = currentState.signedConstantScalar;
        instructionOperand.optSign = currentState.optSign;
        currentState.operands.push_back(instructionOperand);
    }
    
    void add_instResult(){
        LOG_TRACE("Adding instResult")
        //since we are assuming one result we just store sutff that may are overwritten by an operand
        //so I do not store optFaceType, optColorType and optWriteMask
        instructionResult_t instructionResult;
        instructionResult.type = currentState.instResult;
        instructionResult.instResultBase = currentState.instResultBase;
        instructionResult.optWriteMask = currentState.optWriteMask;
        instructionResult.optArrayMem = currentState.optArrayMem;
        instructionResult.arrayMemAbs = currentState.arrayMemAbs;
        instructionResult.optArrayMemAbs = currentState.optArrayMemAbs;
        instructionResult.optColorType = currentState.optColorType;
        instructionResult.optFaceType = currentState.optFaceType;
        instructionResult.resultBasic = currentState.resultBasic;
        instructionResult.faceType = currentState.faceType;
        instructionResult.colorType = currentState.colorType;
        instructionResult.identifier = currentState.identifier;
        currentState.destinations.push_back(instructionResult);
    }
    
    void handler_instOperand(){
        HANDLE_TRACE(currentState.operands[passedData.requestIndexNumber].type)
        (this->*functionCallsTable[currentState.operands[passedData.requestIndexNumber].type])();
    }
    
    void handler_instOperandV(){
        std::string* instOperandV = 
                &currentState.operands[passedData.requestIndexNumber].instOperandV;
        HANDLE_TRACE(*instOperandV)
        (this->*functionCallsTable[*instOperandV])();
    }

    union IntFloat {
        uint32_t i;
        float f;
    };    
    
    void handler_instOperandBaseV(){
        std::string* instOperandBaseV = 
                &currentState.operands[passedData.requestIndexNumber].instOperandBaseV;
        HANDLE_TRACE(*instOperandBaseV)
        (this->*functionCallsTable[*instOperandBaseV])();
        
        if(currentState.operands[passedData.requestIndexNumber].optSign){
            //if the value is hex we need to convert the number to negative instead of appending '-'
            //note: might be moved to another place to provide unified sign handling
            if (passedData.resultValue.substr(0, 2) == "0f") {
                std::string hexNum = passedData.resultValue.substr(2, passedData.resultValue.size()-2);
                IntFloat value;
                value.i = strtoul(hexNum.c_str(),NULL,16);
                value.f = value.f *-1;
                std::stringstream ss;
                ss.setf(std::ios::fixed,std::ios::floatfield);
                ss<<"0f"<<std::setfill('0')<<std::setw(8)<<std::hex<<value.i;
                passedData.resultValue = ss.str();
            } else
                if (passedData.resultValue[0] == '-') passedData.resultValue = passedData.resultValue.substr(1, passedData.resultValue.size() - 1);
            else passedData.resultValue = "-" + passedData.resultValue;
        }
    }
    
    void handler_tempAttribParamBufferUseV(){
        std::string* tempAttribParamBufferUseV = 
                &currentState.operands[passedData.requestIndexNumber].tempAttribParamBufferUseV;
        HANDLE_TRACE(*tempAttribParamBufferUseV)
        (this->*functionCallsTable[*tempAttribParamBufferUseV])();
    }
    
    void handler_paramVarName(){
        instructionOperand_t operand = currentState.operands[passedData.requestIndexNumber];
        HANDLE_TRACE(operand.identifier)   
        passedData.resultValue = operand.identifier + passedData.maskChar;
        //need to handle index here
        //if(operand.optArrayMem)
        //    resultOperand = operand.identifier;
    }
    
    void handler_attribUseV(){
        std::string* attribUseV = 
                &currentState.operands[passedData.requestIndexNumber].attribUseV;
        HANDLE_TRACE(*attribUseV)
        (this->*functionCallsTable[*attribUseV])();
    }
    
    void addRelatedAttribData(std::string varName, int attribIndex){
        unsigned int attribIndexShift;
        std::stringstream attribIndexShitftStr;
        std::string corresspondantLoadOperation;
        std::stringstream defOperation;
        attribIndexShift = attribIndex*constants.INT_VECTOR_SIZE_IN_BYTES; //pointer to vector attrib index in bytes
        attribIndexShift+= passedData.maskCharIntValue * constants.INT_DATA_TYPE_SIZE; //index of the element in the vector in bytes
        assert(attribIndexShift<=2147483647 and attribIndexShift>=-2147483648);
        attribIndexShitftStr<<attribIndexShift;            
        corresspondantLoadOperation = "ld.global.f32 "+varName+", [p_vertex_data+"+attribIndexShitftStr.str()+"];";
        defOperation<<".reg .f32 "<<varName<<";";
        addIfDoesNotExist(&upperInstructionsList,defOperation.str());
        addIfDoesNotExist(&upperInstructionsList,corresspondantLoadOperation);
        passedData.resultValue = varName;
    }
    
    void handler_attribBasic(){
        instructionOperand_t operand = currentState.operands[passedData.requestIndexNumber];
        HANDLE_TRACE(operand.attribBasic.type+"."+ operand.attribBasic.name)
        //TO DO: temporary implementation for testing
        passedData.resultValue = "NOT_DEFINED";
        std::stringstream attribIndexShitftStr;
        std::string corresspondantLoadOperation;
        unsigned int attribIndexShift;
        std::stringstream varName;
        std::stringstream defOperation;
        int attribIndex;
        if(currentState.program_type=="VERTEX_PROGRAM"){
            varName<<"vertex_";
            if(operand.attribBasic.name=="position"){
                varName<<"position_"<<passedData.maskChar;
                attribIndex=VERT_ATTRIB_POS;
            } else if(operand.attribBasic.name=="texcoord"){
                int texCoordIndex=0;
                if(operand.optArrayMem) texCoordIndex=operand.arrayMemAbs;
                varName<<"texcoord"<<texCoordIndex<<passedData.maskChar;
                attribIndex = texCoordIndex+VERT_ATTRIB_TEX0;
            } else if(operand.attribBasic.name=="normal"){
                varName<<"normal_"<<passedData.maskChar;
                attribIndex = VERT_ATTRIB_NORMAL;
            } else if(operand.attribBasic.name=="attrib"){
                varName<<"attrib"<<operand.arrayMemAbs<<passedData.maskChar;
                attribIndex= operand.arrayMemAbs+VERT_ATTRIB_GENERIC0;
            }
            addRelatedAttribData(varName.str(), attribIndex);
        } else if(currentState.program_type=="FRAGMENT_PROGRAM"){
            varName<<"%fragment_";
            if(operand.attribBasic.name=="texcoord"){
                int texCoordIndex=0;
                if(operand.optArrayMemAbs) texCoordIndex=operand.arrayMemAbs;
                varName<<"texcoord"<<texCoordIndex<<"."<<passedData.maskChar;
            } else if(operand.attribBasic.name=="position"){
                varName<<"position"<<"."<<passedData.maskChar;
            }
        } else abort();
        
        passedData.resultValue = varName.str();
    }
    
    void handler_attribColor(){
       instructionOperand_t operand = currentState.operands[passedData.requestIndexNumber];
       HANDLE_TRACE(operand.attribColor+".color")
       //TO DO only test code should be instead of copied from above 
       assert(!operand.optColorType);
       passedData.resultValue = "NOT_DEFINED";
       std::stringstream varName;

        int attribIndex;
        if(currentState.program_type=="VERTEX_PROGRAM"){
            varName<<"vertex_";
        } else if(currentState.program_type=="FRAGMENT_PROGRAM"){
            abort();
        } else abort();
        
        if(operand.optColorType and operand.colorType=="secondary"){
            varName<<"color1_"<<passedData.maskChar;
            attribIndex = VERT_ATTRIB_COLOR1;
        } else {
            varName<<"color0_"<<passedData.maskChar;
            attribIndex = VERT_ATTRIB_COLOR0;   
        }
        addRelatedAttribData(varName.str(),attribIndex);
    }
    
    void handler_namingStatement(){
        HANDLE_TRACE(currentState.namingStatement)
        (this->*functionCallsTable[currentState.namingStatement])();
    }
    
    void handler_ATTRIB_statement(){
        //might redo use the operand trace state (after being merged?)
        HANDLE_TRACE("ATTRIB "+currentState.identifier)
        mappedVar_t mappedVar;
        mappedVar.mappingType = "ATTRIB";
        mappedVar.optArrayMemAbs = false;
        mappedVar.optArrayRange = false;
        if(currentState.attribUseD=="attribBasic"){
            mappedVar.type = currentState.attribBasic.type;
            mappedVar.name = currentState.attribBasic.name;
            if( (mappedVar.name=="clip" || mappedVar.name=="attrib") 
                || ((mappedVar.name == "texcoord" || mappedVar.name == "weight") && currentState.optArrayMemAbs)){
                //even if this is implied for clip and attrib, however we use optArrayMemAbs in generating upon
                //mapping the variables in mapOperand
                mappedVar.optArrayMemAbs = true; 
                mappedVar.arrayMemAbs = currentState.arrayMemAbs;
            }
        } else if (currentState.attribUseD=="attribColor"){
            mappedVar.type = currentState.attribColor;
            mappedVar.name = "color";
            mappedVar.optColorType = currentState.optColorType;
            mappedVar.colorType = currentState.colorType;
        } else if (currentState.attribUseD=="attribMulti"){
            if(mappedVar.type == "attribClip"){mappedVar.type = "fragment"; mappedVar.name = "clip";}
            if(mappedVar.type == "attribTexCoord"){mappedVar.type = currentState.attribTexCoord; mappedVar.name = "texcoord";}
            if(mappedVar.type == "attribGeneric"){mappedVar.type = currentState.attribGeneric;mappedVar.name = "attrib";}
            mappedVar.optArrayRange = true;
            mappedVar.arrayRange_start= currentState.arrayRange.start;
            mappedVar.arrayRange_end = currentState.arrayRange.end;
        } else {std::cout<<"Error: attribUseD type undefined\n"; abort();}
        
        variablesMappingTable[currentState.identifier]=mappedVar;
    }
    
    void handler_OUTPUT_statement(){
        HANDLE_TRACE("OUTPUT "+ currentState.identifier);
        mappedVar_t mappedVar;
        mappedVar.mappingType = "OUTPUT";
        mappedVar.optArrayMemAbs = false;
        mappedVar.optArrayRange = false;
        if(currentState.resultUseD=="resultBasic"){
            mappedVar.name = currentState.resultBasic;
            mappedVar.type = currentState.resultBasic;
            if(mappedVar.name=="attrib" || mappedVar.name=="clip" || (mappedVar.name=="texcoord" && currentState.optArrayMemAbs)){
                //even if this is implied for clip and attrib, however we use optArrayMemAbs in generating upon
                //mapping the variables in mapResult
                mappedVar.optArrayMemAbs = true;
                mappedVar.arrayMemAbs = currentState.arrayMemAbs;
            }
        } else if (currentState.resultUseD=="resultMulti"){
            mappedVar.name = currentState.resultMulti;
            mappedVar.type = currentState.resultMulti;
            mappedVar.optArrayRange = true;
            mappedVar.arrayRange_start = currentState.arrayRange.start;
            mappedVar.arrayRange_end = currentState.arrayRange.end;
        } else {std::cout<<"Error: resultUseD type undefined\n"; abort();}
        
        mappedVar.optFaceType = currentState.optFaceType;
        mappedVar.faceType = currentState.faceType;
        mappedVar.optColorType = currentState.optColorType;
        mappedVar.colorType = currentState.colorType;
        
        variablesMappingTable[currentState.identifier]=mappedVar;
    }
    
    void handler_PARAM_statement(){
        HANDLE_TRACE("PARAM "+currentState.identifier)
        mappedVar_t mappedVar;
        mappedVar.mappingType= "PARAM";
        mappedVar.paramMultInitList = currentState.paramMultInitList;
        
        //need to handle the case of PARAM_singleStmt
        //start here handle the singleStmt
        int currentIndex =0;
        for(int i=0; i<currentState.paramMultInitList.size();i++){
            arrayRange_t range;
            range.start = currentIndex;
            int length = currentState.paramMultInitList[i].arrayRange.end
            - currentState.paramMultInitList[i].arrayRange.start;
            
            if((currentState.PARAM_statement=="PARAM_multipleStmt")
                and ((currentState.paramMultInitList[i].paramUseDM=="stateMatrixRows"
                     and currentState.paramMultInitList[i].optRowArrayRange)
                    or (currentState.paramMultInitList[i].paramUseDM=="programMultipleItem" 
                     and (currentState.paramMultInitList[i].progEnvParams=="arrayRange"
                     or currentState.paramMultInitList[i].progLocalParams=="arrayRange" )))){
                range.end = currentIndex+length;
                currentIndex+=length;
            } else range.end = currentIndex;            
            
             currentIndex++;
             mappedVar.paramValuesRanges.push_back(range);
        }        
        
        variablesMappingTable[currentState.identifier] = mappedVar;
        assert(mappedVar.paramMultInitList.size()==mappedVar.paramValuesRanges.size());
    }
    
    void handler_TEMP_statement(){
        HANDLE_TRACE("TEMP variables list")
        std::stringstream ss;
        for(int i=0; i<currentState.varNameList.size();i++){
            ss<<".reg";
            if(currentState.varMods)
                ss<<" "<<modifiersMappingTable[currentState.varModifier];
            else ss<<" "<<modifiersMappingTable["FLOAT"];
            
            ss<<" "<<currentState.varNameList[i]<<"x";
            ss<<", "<<currentState.varNameList[i]<<"y";
            ss<<", "<<currentState.varNameList[i]<<"z";
            ss<<", "<<currentState.varNameList[i]<<"w";
            ss<<";";
        }
        upperInstructionsList.push_back(ss.str());
    }
    
    void handler_stateSingleItem(){
        instructionOperand_t operand = currentState.operands[passedData.requestIndexNumber];
        HANDLE_TRACE(operand.stateSingleItem)
        (this->*functionCallsTable[operand.stateSingleItem])();
    }
    
    void handler_stateLightItem(){
        instructionOperand_t operand = currentState.operands[passedData.requestIndexNumber];
        HANDLE_TRACE(operand.stateLightProperty)
        std::stringstream ss;
        if(operand.stateLightProperty=="position"){
            ss<<"state_light"<<operand.arrayMemAbs<<"_position"<<passedData.maskChar;
            int indexOfRowElement = operand.arrayMemAbs*constants.INT_VECTOR_SIZE_IN_BYTES;
            indexOfRowElement+= passedData.maskCharIntValue*constants.INT_DATA_TYPE_SIZE;
            std::stringstream correspondantLoadOperantion;
            correspondantLoadOperantion<<"ld.const.f32 "<<ss.str()<<", [state_light_position+"<<indexOfRowElement<<"];";
            std::stringstream defOperation;
            defOperation<<".reg .f32 "<<ss.str()<<";";
            addIfDoesNotExist(&upperInstructionsList,defOperation.str());
            addIfDoesNotExist(&upperInstructionsList,correspondantLoadOperantion.str());
        }
        passedData.resultValue = ss.str();
    }
    
    void handler_stateMatrixRow(){
        instructionOperand_t operand = currentState.operands[passedData.requestIndexNumber];
        HANDLE_TRACE(operand.stateMatrixName)
        //I need to check the modifier too
        std::stringstream ss;
        if(operand.stateMatrixName=="mvp"){
            ss<<"state_matrix_mvp_row"<<operand.arrayMemAbs<<passedData.maskChar;
            int indexOfRowElement = operand.arrayMemAbs*constants.INT_VECTOR_SIZE_IN_BYTES;
            indexOfRowElement+=passedData.maskCharIntValue*constants.INT_DATA_TYPE_SIZE;
            std::stringstream correspondantLoadOperantion;
            correspondantLoadOperantion<<"ld.const.f32 "<<ss.str()<<", [state_matrix_mvp+"<<indexOfRowElement<<"];";
            std::stringstream defOperation;
            defOperation<<".reg .f32 "<<ss.str()<<";";
            addIfDoesNotExist(&upperInstructionsList,defOperation.str());
            addIfDoesNotExist(&upperInstructionsList,correspondantLoadOperantion.str());
        }
        else if(operand.stateMatrixName=="modelview" or operand.stateMatrixName=="projection" ){
            std::string mat_name = operand.stateMatrixName;
            ss<<"state_matrix_"<<mat_name;
            if(operand.optStateMatModifier){
                if(operand.stateMatModifier=="inverse")
                    ss<<"_inverse";
                else if(operand.stateMatModifier=="transpose")
                    ss<<"_transpose";
                else if(operand.stateMatModifier=="invtrans")
                    ss<<"_invtrans";
                else abort();
            }
            ss<<"_row"<<operand.arrayMemAbs<<passedData.maskChar;
            int indexOfRowElement = operand.arrayMemAbs*constants.INT_VECTOR_SIZE_IN_BYTES;
            indexOfRowElement+=passedData.maskCharIntValue*constants.INT_DATA_TYPE_SIZE;
            std::stringstream correspondantLoadOperantion;
            if(operand.optStateMatModifier){
                if(operand.stateMatModifier=="inverse")
                    correspondantLoadOperantion<<"ld.const.f32 "<<ss.str()<<", [state_matrix_"<<mat_name<<"_inverse+"<<indexOfRowElement<<"];";
                else if(operand.stateMatModifier=="transpose")
                    correspondantLoadOperantion<<"ld.const.f32 "<<ss.str()<<", [state_matrix_"<<mat_name<<"_inverse+"<<indexOfRowElement<<"];";
                else if(operand.stateMatModifier=="invtrans")
                    correspondantLoadOperantion<<"ld.const.f32 "<<ss.str()<<", [state_matrix_"<<mat_name<<"_inverse+"<<indexOfRowElement<<"];";
                else abort();
            }
            std::stringstream defOperation;
            defOperation<<".reg .f32 "<<ss.str()<<";";
            addIfDoesNotExist(&upperInstructionsList,defOperation.str());
            addIfDoesNotExist(&upperInstructionsList,correspondantLoadOperantion.str());
        } else {
            printf("ARB_to_PTX: Unsupported stateMatrixName \n");
            abort();
        }
        
        passedData.resultValue = ss.str();
    }
    
    void handler_programSingleItem(){
        instructionOperand_t operand = currentState.operands[passedData.requestIndexNumber];
        HANDLE_TRACE(operand.programSingleItem)
        (this->*functionCallsTable[operand.programSingleItem])();
    }
    
    void handler_progLocalParam(){
        //NOTE: uniforms are handled as prog local params as cgc ARB specifies
        instructionOperand_t operand = currentState.operands[passedData.requestIndexNumber];
        std::stringstream paramName;
        paramName<<"progLocalParam #"<<operand.arrayMemAbs;
        HANDLE_TRACE(paramName.str())
                
        
        std::stringstream ss;
        int usedIndex;
        if(startIndexOne)
            usedIndex = operand.arrayMemAbs-1; //only with arbvp1 and arbfp1
        else usedIndex = operand.arrayMemAbs;
        
        if(usedIndex<0) usedIndex = 0; //in this case 0 index is actually uses, that is what was observed!
        ss<<"program_local"<<usedIndex<<passedData.maskChar;
        int indexOfRowElement = usedIndex*constants.INT_VECTOR_SIZE_IN_BYTES;
        indexOfRowElement+=passedData.maskCharIntValue*constants.INT_DATA_TYPE_SIZE;
        std::stringstream correspondantLoadOperantion;
        correspondantLoadOperantion<<"ld.const.f32 "<<ss.str()<<", ["<<localParamsName<<"+"<<indexOfRowElement<<"];";
        std::stringstream defOperation;
        defOperation<<".reg .f32 "<<ss.str()<<";";
        addIfDoesNotExist(&upperInstructionsList,defOperation.str());
        addIfDoesNotExist(&upperInstructionsList,correspondantLoadOperantion.str());
        
        passedData.resultValue = ss.str();
    }
    
    void handler_constantVector(){
        instructionOperand_t operand = currentState.operands[passedData.requestIndexNumber];
        std::stringstream ss;
        for(int i=0;i<operand.constantVectorList.size();i++)
            ss<<operand.constantVectorList[i]<<", ";
        HANDLE_TRACE(ss.str().substr(0,ss.str().size()-2));
        ss.str("");
        ss.setf(std::ios::fixed,std::ios::floatfield);
        ss<<"0f"<<std::setfill('0')<<std::setw(8)<<std::hex<<*(int *)&operand.constantVectorList[passedData.maskCharIntValue];
        //ss<<operand.constantVectorList[passedData.maskCharIntValue];
        passedData.resultValue = ss.str();
    }
    
    void handler_signedConstantScalar(){
        instructionOperand_t operand = currentState.operands[passedData.requestIndexNumber];
        HANDLE_TRACE(operand.signedConstantScalar)
        abort();
    }
};

void arbToPtxConverter::initializeFunctionCallsTable(){
    functionCallsTable["statement"] = &arbToPtxConverter::handler_statement;
    functionCallsTable["instruction"] = &arbToPtxConverter::handler_instruction;
    functionCallsTable["namingStatement"] = &arbToPtxConverter::handler_namingStatement;
    functionCallsTable["ALUInstruction"] = &arbToPtxConverter::handler_ALUInstruction;
    functionCallsTable["TexInstruction"] = &arbToPtxConverter::handler_TexInstruction;
    functionCallsTable["VECTORop_instruction"] = &arbToPtxConverter::handler_VECTORop_instruction;
    functionCallsTable["BINop_instruction"] = &arbToPtxConverter::handler_BINop_instruction;
    functionCallsTable["TRIop_instruction"] = &arbToPtxConverter::handler_TRIop_instruction;
    functionCallsTable["SCALARop_instruction"] = &arbToPtxConverter::handler_SCALARop_instruction;
    functionCallsTable["instResult"] = &arbToPtxConverter::handler_instResult;
    functionCallsTable["instResultBase"] = &arbToPtxConverter::handler_instResultBase;
    functionCallsTable["resultVarName"] = &arbToPtxConverter::handler_resultVarName;
    functionCallsTable["instOperandBaseV"] = &arbToPtxConverter::handler_instOperandBaseV;
    functionCallsTable["tempAttribParamBufferUseV"] = &arbToPtxConverter::handler_tempAttribParamBufferUseV;
    functionCallsTable["paramVarName"] = &arbToPtxConverter::handler_paramVarName;
    functionCallsTable["instOperandV"] = &arbToPtxConverter::handler_instOperandV;
    functionCallsTable["TEMP_statement"] = &arbToPtxConverter::handler_TEMP_statement;
    functionCallsTable["ATTRIB_statement"] = &arbToPtxConverter::handler_ATTRIB_statement;
    functionCallsTable["PARAM_statement"] = &arbToPtxConverter::handler_PARAM_statement;
    functionCallsTable["instOperand"] = &arbToPtxConverter::handler_instOperand;
    functionCallsTable["OUTPUT_statement"] = &arbToPtxConverter::handler_OUTPUT_statement;
    functionCallsTable["attribUseV"] = &arbToPtxConverter::handler_attribUseV;
    functionCallsTable["attribBasic"] = &arbToPtxConverter::handler_attribBasic;
    functionCallsTable["attribColor"] = &arbToPtxConverter::handler_attribColor;
    functionCallsTable["resultBasic"] = &arbToPtxConverter::handler_resultBasic;
    functionCallsTable["stateSingleItem"] = &arbToPtxConverter::handler_stateSingleItem;
    functionCallsTable["programSingleItem"] = &arbToPtxConverter::handler_programSingleItem;
    functionCallsTable["constantVector"] = &arbToPtxConverter::handler_constantVector;
    functionCallsTable["signedConstantScalar"] = &arbToPtxConverter::handler_signedConstantScalar;
    functionCallsTable["stateMatrixRow"] = &arbToPtxConverter::handler_stateMatrixRow;
    functionCallsTable["stateLightItem"] = &arbToPtxConverter::handler_stateLightItem;
    functionCallsTable["TEXop_instruction"] = &arbToPtxConverter::handler_TEXop_instruction;
    functionCallsTable["progLocalParam"] = &arbToPtxConverter::handler_progLocalParam;
}

void arbToPtxConverter::initializeInstructionMappingTable(){
    instructionMap_t MOV;
    MOV.type = instructionMap_t::DirectVector2Scalar;
    MOV.ptxInstruction = "mov";
    instructionMappingTable["MOV"]= MOV;
    
    instructionMap_t ADD;
    ADD.type = instructionMap_t::DirectVector2Scalar;
    ADD.ptxInstruction = "add";
    instructionMappingTable["ADD"]= ADD;
    
    instructionMap_t MUL;
    MUL.type = instructionMap_t::DirectVector2Scalar;
    MUL.ptxInstruction = "mul";
    instructionMappingTable["MUL"]= MUL;
    
    instructionMap_t MAD;
    MAD.type = instructionMap_t::DirectVector2Scalar;
    MAD.ptxInstruction = "mad";
    instructionMappingTable["MAD"]= MAD;
    
    instructionMap_t DP3;
    DP3.type = instructionMap_t::UnDirectVector2Scalar;
    instructionMappingTable["DP3"]=DP3;
    
    instructionMap_t DP4;
    DP4.type = instructionMap_t::UnDirectVector2Scalar;
    instructionMappingTable["DP4"]=DP4;
    
    instructionMap_t TEX;
    TEX.type = instructionMap_t::UnDirectVector2Scalar;
    instructionMappingTable["TEX"]=TEX;
    
    instructionMap_t RSQ;
    RSQ.type = instructionMap_t::DirectVector2Scalar;
    RSQ.ptxInstruction = "rsqrt.approx";
    instructionMappingTable["RSQ"] = RSQ;
    
    instructionMap_t SEQ;
    SEQ.type = instructionMap_t::DirectVector2Scalar;
    SEQ.ptxInstruction = "set.eq.f32";
    instructionMappingTable["SEQ"] = SEQ;
    
    instructionMap_t SGE;
    SGE.type = instructionMap_t::DirectVector2Scalar;
    SGE.ptxInstruction = "set.ge.f32";
    instructionMappingTable["SGE"] = SGE;
    
    instructionMap_t SGT;
    SGT.type = instructionMap_t::DirectVector2Scalar;
    SGT.ptxInstruction = "set.gt.f32";
    instructionMappingTable["SGT"] = SGT;
    
    instructionMap_t SLE;
    SLE.type = instructionMap_t::DirectVector2Scalar;
    SLE.ptxInstruction = "set.le.f32";
    instructionMappingTable["SLE"] = SLE;
    
    instructionMap_t SLT;
    SLT.type = instructionMap_t::DirectVector2Scalar;
    SLT.ptxInstruction = "set.lt.f32";
    instructionMappingTable["SLT"] = SLT;
    
    instructionMap_t SNE;
    SNE.type = instructionMap_t::DirectVector2Scalar;
    SNE.ptxInstruction = "set.ne.f32";
    instructionMappingTable["SNE"] = SNE;
    
    instructionMap_t ABS;
    ABS.type = instructionMap_t::DirectVector2Scalar;
    ABS.ptxInstruction = "abs";
    instructionMappingTable["ABS"] = ABS;
    
    instructionMap_t FLR;
    FLR.type = instructionMap_t::DirectVector2Scalar;
    FLR.ptxInstruction = "frc";
    instructionMappingTable["FLR"] = FLR;
    
    instructionMap_t MAX;
    MAX.type = instructionMap_t::DirectVector2Scalar;
    MAX.ptxInstruction = "max";
    instructionMappingTable["MAX"] = MAX;
    
    instructionMap_t CMP;
    CMP.type = instructionMap_t::UnDirectVector2Scalar;
    instructionMappingTable["CMP"] = CMP;
}

std::string arbToPtxConverter::getPTXVariable(std::string pType, int index,char pMask, int pMaskIndex){
    HANDLE_TRACE(pType)
    assert(index >=0 && index <=3);
    passedData.requestIndexNumber = index;
    passedData.maskChar = pMask;
    passedData.maskCharIndex = pMaskIndex;        
    if(pType=="instResult"){
        passedData.resultWriteMaskIntValue= getMaskIntegerValue(pMask);
        mapResult(index);    
        (this->*functionCallsTable["instResult"])();
    } else if(pType=="instOperand"){
        mapOperand(index); 
        instructionOperand_t operand = currentState.operands[passedData.requestIndexNumber];
        assert(passedData.requestIndexNumber<operand.swizzleSuffix.size());
        passedData.maskChar = operand.swizzleSuffix[getMaskIntegerValue(passedData.maskChar)];
        passedData.maskCharIntValue = getMaskIntegerValue(passedData.maskChar);
        (this->*functionCallsTable["instOperand"])();
    }
    
    return passedData.resultValue;// + passedData.maskChar;
}

void arbToPtxConverter::mapOperand(int index){
    HANDLE_TRACE("the mapping of \""+currentState.operands[index].identifier+"\"")
    if(variablesMappingTable.find(currentState.operands[index].identifier)==variablesMappingTable.end())
        return;
   
    //if attrib is redone to be an operand type this will be just copying corresponding fields 
    mappedVar_t var = variablesMappingTable[currentState.operands[index].identifier];
    instructionOperand_t* operandP = &currentState.operands[index];
    operandP->identifier = ""; //no need to map operand twice
    
    if(var.mappingType == "ATTRIB"){
        operandP->instOperandBaseV = "attribUseV";
        //operandP->identifier = var.type + "." + var.name;
        operandP->attribBasic.name = var.name;
        operandP->attribBasic.type = var.type;
        operandP->attribUseV = "attribBasic";
        if(var.name=="color")
            operandP->attribUseV = "attribColor";
        operandP->optColorType = var.optColorType;
        operandP->colorType = var.colorType;
        
        //need to check that  zzzz TO DO
        //they do not go out of range abs+range_start < range
        //that the original does have index while the mapping not e.g. bla[2] which only maps vertex.attrib
        //also stuff down sounds redundant code
        if(var.optArrayRange) operandP->arrayMemAbs+= var.arrayRange_start;
        else if(var.optArrayMemAbs) operandP->arrayMemAbs = var.arrayMemAbs;
        if((var.optArrayMemAbs||var.optArrayRange) && !operandP->optArrayMem){
            operandP->optArrayMemAbs=true;
            operandP->arrayMem = "arrayMemAbs";
            operandP->arrayMemAbs = var.optArrayMemAbs? var.arrayMemAbs :var.arrayRange_start;
        }
    }
    
    if(var.mappingType == "PARAM"){
        int paramIndex = 0;
        if(operandP->optArrayMem) 
            paramIndex = operandP->arrayMemAbs;
        
        paramMultInitList_t * paramVar = NULL;
        arrayRange_t paramRange;
        assert(var.paramMultInitList.size()==var.paramValuesRanges.size()); //they should have the same size
        for(int i=0;i<var.paramMultInitList.size();i++){
            if(paramIndex>=var.paramValuesRanges[i].start and paramIndex<=var.paramValuesRanges[i].end){
                paramVar = &var.paramMultInitList[i];
                paramRange = var.paramValuesRanges[i];
                break;
            }
        }
        assert(paramVar!=NULL); 
        operandP->constantVectorList = paramVar->constantVectorList;
        operandP->signedConstantScalar = paramVar->signedConstantScalar;
        operandP->programSingleItem = paramVar->programSingleItem;
        operandP->stateSingleItem = paramVar->stateSingleItem;
        operandP->optFaceType = paramVar->optFaceType;
        operandP->faceType = paramVar->faceType;
        operandP->stateMatProperty = paramVar->stateMatProperty;
        operandP->stateLightProperty = paramVar->stateLightProperty;
        operandP->stateLModProperty = paramVar->stateLModProperty;
        operandP->stateLProdProperty = paramVar->stateLProdProperty;
        operandP->stateFogProperty = paramVar->stateFogProperty;
        operandP->optStateMatModifier = paramVar->optStateMatModifier;
        operandP->stateMatModifier = paramVar->stateMatModifier;
        operandP->stateMatrixName = paramVar->stateMatrixName;
        operandP->stateTexGenType = paramVar->stateTexGenType;
        operandP->stateTexGenCoord = paramVar->stateTexGenCoord;
        operandP->statePointProperty = paramVar->statePointProperty;
        
        std::string tempType;
        if(paramVar->type=="paramUseDB"){
            tempType=paramVar->paramUseDB;
            operandP->optArrayMemAbs = paramVar->optArrayMemAbs;
            operandP->arrayMemAbs = paramVar->arrayMemAbs;
        }
        else if (paramVar->type=="paramUseDM"){
            tempType=paramVar->paramUseDM;
            tempType = tempType=="stateMatrixRows"? "stateSingleItem":tempType;
            tempType = tempType=="programMultipleItem"? "programSingleItem":tempType;
            
            int indexOfParam =0;
            if(paramVar->paramUseDM=="stateMatrixRows"){
                if(paramVar->optRowArrayRange)  indexOfParam = paramVar->arrayRange.start;
                if(operandP->optArrayMem)       indexOfParam+= operandP->arrayMemAbs - paramRange.start;
                operandP->stateSingleItem = "stateMatrixRow";
            } else if(paramVar->paramUseDM=="programMultipleItem"){
                operandP->programSingleItem = paramVar->programMultipleItem.substr(0,paramVar->programMultipleItem.size()-1);
                if((paramVar->progLocalParams=="arrayRange" and paramVar->programMultipleItem=="progLocalParams")
                    or (paramVar->progEnvParams=="arrayRange" and paramVar->programMultipleItem=="progEnvParams")){
                    indexOfParam= paramVar->arrayRange.start;
                    if(operandP->optArrayMem)       indexOfParam+= operandP->arrayMemAbs - paramRange.start;
                } else if((paramVar->progLocalParams=="arrayMemAbs" and paramVar->programMultipleItem=="progLocalParams")
                    or (paramVar->progEnvParams=="arrayMemAbs" and paramVar->programMultipleItem=="progEnvParams"))
                    indexOfParam = paramVar->arrayMemAbs;
            }
            operandP->arrayMemAbs = indexOfParam;
        }
        else {printf("Unidentified PARAM type \n"); 
        abort();}
        
        operandP->tempAttribParamBufferUseV = tempType;
    }
}

void arbToPtxConverter::mapResult(int index){
    HANDLE_TRACE("the mapping of \""+currentState.destinations[index].identifier+"\"")
    if(variablesMappingTable.find(currentState.destinations[index].identifier)==variablesMappingTable.end())
        return;
    mappedVar_t var = variablesMappingTable[currentState.destinations[index].identifier];
    assert(var.mappingType != "ATTRIB");
    assert(var.mappingType != "PARAM");
    instructionResult_t* resultP = &currentState.destinations[index];
    resultP->identifier = ""; //only we need to map once
    if(var.mappingType=="OUTPUT"){
        resultP->instResultBase = "resultBasic";
        resultP->resultBasic = var.name;
        resultP->optFaceType = var.optFaceType;
        resultP->faceType = var.faceType;
        resultP->optColorType = var.optColorType;
        resultP->colorType = var.colorType;
        
        //need the same checks as indicated above in mapOperand regarding some cases
        if(var.optArrayRange) resultP->arrayMemAbs+= var.arrayRange_start;
        else if (var.optArrayMemAbs) resultP->arrayMemAbs = var.arrayMemAbs;
        if((var.optArrayMemAbs || var.optArrayRange) && !resultP->optArrayMem){
            resultP->optArrayMemAbs = true;
            resultP->arrayMemAbs = var.optArrayMemAbs? var.arrayMemAbs : var.arrayRange_start;
        }
    }
}

void arbToPtxConverter::initializeModifiersMappingTable(){
    modifiersMappingTable["F"] = ".f32";
    modifiersMappingTable["U"] = ".u32";
    modifiersMappingTable["SHORT"] = ".s16";
    modifiersMappingTable["LONG"] = ".s64";
    modifiersMappingTable["INT"] = ".s32";
    modifiersMappingTable["UINT"] = ".u32";
    modifiersMappingTable["FLOAT"] = ".f32";
}

void arbToPtxConverter::loadCommonShaderData(){
    upperInstructionsList.push_back(".version 2.0");
    upperInstructionsList.push_back(".target sm_10, map_f64_to_f32");
    upperInstructionsList.push_back(".const .align 4 .f32 state_matrix_mvp[16];");
    upperInstructionsList.push_back(".const .align 4 .f32 state_matrix_modelview_inverse[16];");
    upperInstructionsList.push_back(".const .align 4 .f32 state_matrix_projection_inverse[16];");
    std::stringstream lightsSize;
    lightsSize<<MAX_LIGHTS*constants.INT_VECTOR_SIZE;
    upperInstructionsList.push_back(".const .align 4 .f32 state_light_position["+lightsSize.str()+"];");
    std::stringstream maxUniformsCount;
    maxUniformsCount<<MAX_UNIFORMS*constants.INT_VECTOR_SIZE;
    upperInstructionsList.push_back(".const .align 4 .f32 vertex_program_locals["+maxUniformsCount.str()+"];");
    upperInstructionsList.push_back(".const .align 4 .f32 fragment_program_locals["+maxUniformsCount.str()+"];");
    
    
}
void arbToPtxConverter::initializeFragmentShaderData(){
    
    localParamsName = "fragment_program_locals";
    std::string functionName = ".entry "+shaderFuncName+" (.param .u64 __cudaparm_"+shaderFuncName+"_outputData){";
    std::string dataPointer = "p_result_data";
    std::string depthPointer = "p_depth_data";
    std::string paramName = "__cudaparm_"+shaderFuncName+"_outputData";
    upperInstructionsList.push_back(functionName);
    upperInstructionsList.push_back(".reg .pred pexit;");
    upperInstructionsList.push_back("setp.eq.u32 pexit, 0, %fragment_active;");
    upperInstructionsList.push_back("@pexit exit;");
    upperInstructionsList.push_back(".reg .u64 "+dataPointer+";");
    upperInstructionsList.push_back("ld.param.u64 "+dataPointer+", ["+paramName+"];");
    upperInstructionsList.push_back(".reg .f32 tempReg0,tempReg1, tempReg2, tempReg3;");

    upperInstructionsList.push_back(".reg .u64 fx, fy;");
    upperInstructionsList.push_back("cvt.rzi.u64.f32 fx, %fragment_position.x;");
    //upperInstructionsList.push_back("printf.u64 3, fx;");
    upperInstructionsList.push_back("shl.b64 fx,fx,2;");
    upperInstructionsList.push_back("cvt.rzi.u64.f32 fy, %fragment_position.y;");
    //upperInstructionsList.push_back("printf.u64 4, fy;");
    upperInstructionsList.push_back("mul.lo.u64 fy,fy,%rb_width;");
    upperInstructionsList.push_back("shl.b64 fy,fy,2;");
    upperInstructionsList.push_back("sub.u64 p_result_data,p_result_data,fy;");
    upperInstructionsList.push_back("add.u64 p_result_data,p_result_data,fx;");

    if(depthEnabled){
       upperInstructionsList.push_back(".reg .u64 "+depthPointer+", ur2;");
       if(depthSize == "Z16"){
          upperInstructionsList.push_back("shl.b64 ur2,%rb_size,1;");
          upperInstructionsList.push_back("shr.b64 fx,fx,1;");
          upperInstructionsList.push_back("shr.b64 fy,fy,1;");

          //upperInstructionsList.push_back("printf.u64 8, %rb_size;");
          //upperInstructionsList.push_back("printf.u64 5, fx;");
          //upperInstructionsList.push_back("printf.u64 6, fy;");
          //upperInstructionsList.push_back("printf.u64 7, ur2;");
       } else if(depthSize == "Z32"){
          upperInstructionsList.push_back("shl.b64 ur2,%rb_size,2;");
       } 

       upperInstructionsList.push_back("ld.param.u64 "+depthPointer+", ["+paramName+"];");
       //upperInstructionsList.push_back("printf.u64 21, "+depthPointer + ";");
       upperInstructionsList.push_back("add.u64 "+depthPointer+","+depthPointer+",ur2;");
       //upperInstructionsList.push_back("printf.u64 22, "+depthPointer + ";");
       upperInstructionsList.push_back("sub.u64 "+depthPointer+","+depthPointer+",fy;");
       upperInstructionsList.push_back("add.u64 "+depthPointer+","+depthPointer+",fx;");
    }
}

void arbToPtxConverter::initializeVertexShaderData(){
    loadCommonShaderData();
    localParamsName = "vertex_program_locals";
    std::string functionName = ".entry "+shaderFuncName+" (.param .u64 __cudaparm_"+shaderFuncName+"_vertexData){";
    std::string dataPointer = "p_vertex_data";
    std::string paramName = "__cudaparm_"+shaderFuncName+"_vertexData";
    upperInstructionsList.push_back(functionName);
    upperInstructionsList.push_back(".reg .pred pexit;");
    upperInstructionsList.push_back("setp.eq.u32 pexit, 0, %vertex_active;");
    upperInstructionsList.push_back("@pexit exit;");
    upperInstructionsList.push_back(".reg .u64 "+dataPointer+";");
    upperInstructionsList.push_back("ld.param.u64 "+dataPointer+", ["+paramName+"];");
    upperInstructionsList.push_back(".reg .f32 tempReg0,tempReg1, tempReg2, tempReg3;");
    
    upperInstructionsList.push_back(".reg .u64 thread_id64;");
    upperInstructionsList.push_back("mov.u64 thread_id64, %utid;");
    upperInstructionsList.push_back(".reg .u64 vertex_data_index;");
    upperInstructionsList.push_back("mul.wide.u32 vertex_data_index, thread_id64, "+constants.STR_PER_VERTEX_DATA_SIZE+";");
    upperInstructionsList.push_back("add.u64 p_vertex_data, p_vertex_data, vertex_data_index;");
    for(unsigned i=0;i<upperInstructionsList.size();i++)
        generatedInstructions[upperInstructionsList[i]]=true;
}

void arbToPtxConverter::addIfDoesNotExist(std::vector<std::string> * list, std::string instruction){
    if(!generatedInstructions[instruction]){
        list->push_back(instruction);
        generatedInstructions[instruction]=true;
    }
}

unsigned arbToPtxConverter::getMaskIntegerValue(char pMask){
    switch(pMask){
    case 'x': return 0;break;
    case 'y': return 1;break;
    case 'z': return 2;break;
    case 'w': return 3;break;
    default: abort();
    }
}
void arbToPtxConverter::clearState(){
    currentState.operands.clear();
    currentState.opModifierItem.clear();
    currentState.destinations.clear();
    currentState.varNameList.clear();
    currentState.paramMultInitList.clear();
    currentState.identifier = "";
    currentState.satFlag = false;
}

arbToPtxConverter globalGLSLtoPTXObject;
