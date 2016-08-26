
#if defined(_WIN32)

#include <windows.h>

#else

#define PtrToUlong(X)  (long)(X)

#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include <GL/glew.h>

#if __APPLE__
#include <GLUT/glut.h> /* OpenGL Utility Toolkit (GLUT) */
#else
#include <GL/glut.h>   /* OpenGL Utility Toolkit (GLUT) */
#endif

#include <Cg/cg.h>
#include <Cg/cgGL.h>

#include "register_states.h"

typedef enum {
    STATE_TYPE_MIN = 0,
    STATE_TYPE_GENERIC = 0,
    STATE_TYPE_GL,
    STATE_TYPE_MAX,
} StateType;

typedef enum {
    CG_FILE_TYPE_MIN = 0,
    CG_FILE_TYPE_UNKNOWN = 0,
    CG_FILE_TYPE_EFFECT,
    CG_FILE_TYPE_PROGRAM_SOURCE,
    CG_FILE_TYPE_PROGRAM_OBJECT,
    CG_FILE_TYPE_MAX,
} CgFileType;

static void HandleOptions(int argc, char *argv[]);
static void PrintUsage(void);
static CgFileType GuessFileType(char *pFileName);
static void GetProfileAndEntryFromObjectFile(const char *pFileName);

static void InitContext(void);
static void DestroyContext(void);
static void CheckForCgError(const char *situation);

static void tab_increment(void);
static void tab_decrement(void);
static void tab_printf(char* format, ...);
static void tab_printf_enum(char* msg, CGenum e);
static void tab_printf_bool(char* msg, CGbool b);

static void DumpSupportedProfiles(void);
static void DumpProfileOptions(CGprofile profile);
static void DumpFile(const char* pFileName);
static void DumpContext(CGcontext context);
static void DumpProgram(const char* pTitle, CGprogram program, int nProgram);
static void DumpEffect(CGeffect effect, int nEffect);
static void DumpTechnique(CGtechnique technique, int nTechnique);
static void DumpPass(CGpass pass, int nPass);
static void DumpStateAssignment(CGstateassignment sa, int nStateAssignment, int isSSA);
static void DumpAnnotation(CGannotation annotation, int nAnnotation,
                           CGhandle container, int containerType);
static void DumpState(CGstate state, int nState, int isSamplerState);
static void DumpParameter(const char* pTitle, CGparameter parameter, int nParameter,
                          CGhandle container, CGparameter parent);
static void DumpParameterValues(CGparameter parameter);
static void DumpType(const char* pTitle, int index, CGtype type);

const char* myProgramName = "cgfxcat";  /* Program name for messages. */
char*       pFileName = NULL;
CGprofile   myProfile = CG_PROFILE_UNKNOWN;
char*       myProfileString = NULL;
char*       myEntry = NULL;
CGcontext   myCgContext = 0;
CGtechnique myCgTechnique = 0;
StateType   myStateType = STATE_TYPE_GENERIC;
CgFileType  myFileType = CG_FILE_TYPE_UNKNOWN;

/* HandleOptions modifies argv, so we need some fake ones for glutInit */

static int   myGlutArgc   = 1;
static char *myGlutArgv[] = { "cgfxcat" };

int main( int argc, char *argv[] )
{
    HandleOptions(argc, argv);

    if (myStateType==STATE_TYPE_GL) {
        glutInit(&myGlutArgc,myGlutArgv);
    }

    tab_printf("Version: %s\n", cgGetString(CG_VERSION));
    tab_printf_enum("LockingPolicy", cgGetLockingPolicy());
    tab_printf_enum("SemanticCasePolicy", cgGetSemanticCasePolicy());
    tab_printf("ErrorCallback: %p\n", cgGetErrorCallback());
    tab_printf("ErrorHandler: %p\n", cgGetErrorHandler(NULL));

    DumpSupportedProfiles();

    tab_increment();

    DumpFile(pFileName);

    tab_decrement();

    return 0;
}

static void HandleOptions(int argc, char *argv[])
{
    int ok = 1;
    int ii;

    myProgramName = argv[0];

    pFileName = NULL;

    for (ii=1; ii<argc; ++ii) {
        if (!strcmp(argv[ii], "-gl")) {
            myStateType = STATE_TYPE_GL;
            argv[ii] = NULL;
        } else if (!strcmp(argv[ii], "-effect")) {
            myFileType = CG_FILE_TYPE_EFFECT;
            argv[ii] = NULL;
        } else if (!strcmp(argv[ii], "-program")) {
            myFileType = CG_FILE_TYPE_PROGRAM_SOURCE;
            argv[ii] = NULL;
        } else if (!strcmp(argv[ii], "-object")) {
            myFileType = CG_FILE_TYPE_PROGRAM_OBJECT;
            argv[ii] = NULL;
        } else if (!strcmp(argv[ii], "-profile")) {
            ++ii;
            if (myProfileString == NULL && ii<argc) {
                myProfileString = argv[ii];
                myProfile = cgGetProfile(myProfileString);
            } else {
                ok = 0;
            }
        } else if (!strcmp(argv[ii], "-entry")) {
            ++ii;
            if (myEntry == NULL && ii<argc) {
                myEntry = argv[ii];
            } else {
                ok = 0;
            }
        } else if (!pFileName) {
            pFileName = argv[ii];
            if (*pFileName == '-') {
                ok = 0;
            }
            if (myFileType == CG_FILE_TYPE_UNKNOWN) {
                myFileType = GuessFileType(pFileName);
            }
        } else {
            ok = 0;
        }
    }

    if (!ok || !pFileName) {
        PrintUsage();
        exit(1);
    }
}

static CgFileType GuessFileType(char *pFileName)
{
    CGprofile profile = CG_PROFILE_UNKNOWN;

    if (pFileName) {
        const char * lastdot = strrchr ( pFileName, '.' );
        if (lastdot) {
            if (!strcmp(lastdot, ".cgfx")) {
                return CG_FILE_TYPE_EFFECT;
            } else if (!strcmp(lastdot, ".cg")) {
                return CG_FILE_TYPE_PROGRAM_SOURCE;
            } else if (!strcmp(lastdot, ".fx")) {
                return CG_FILE_TYPE_EFFECT;
            } else if (!strcmp(lastdot, ".hlsl")) {
                return CG_FILE_TYPE_PROGRAM_SOURCE;
            } else {
                profile = cgGetProfile(lastdot+1);
                if (profile != CG_PROFILE_UNKNOWN) {
                    return CG_FILE_TYPE_PROGRAM_OBJECT;
                }
            }
        }
    }

    return CG_FILE_TYPE_EFFECT;
}

static void GetProfileAndEntryFromObjectFile(const char *pFileName)
{
    FILE *pFile;
    int   found = 0;
    char  str[1024];
    char *p;
    char *p2;

    if (pFileName) {
        pFile = fopen(pFileName, "r");
        if (pFile) {
            while (fgets(str, 1023, pFile)) {
                if (!strncmp(str, "#profile", 8)) {
                    p = str+8;
                    if (*p) p++;
                    if (*p) {
                        p2 = p;
                        while (*p2) {
                            if (*p2 == '\n') {
                                *p2 = '\0';
                                break;
                            }
                            p2++;
                        }
                        myProfile = cgGetProfile(p);
                        found++;
                        if (found > 1) {
                            break;
                        }
                    }
                } else if (!strncmp(str, "#program", 8)) {
                    p = str+8;
                    if (*p) p++;
                    if (*p) {
                        p2 = p;
                        while (*p2) {
                            if (*p2 == '\n') {
                                *p2 = '\0';
                                break;
                            }
                            p2++;
                        }
                        // this memory will ultimately be leaked, but
                        // for the purposes of this program, so what.
                        myEntry = strdup(p);
                        found++;
                        if (found > 1) {
                            break;
                        }
                    }
                }
            }
            fclose(pFile);
        }
    }
}

static void PrintUsage(void)
{
    fprintf(stderr, "Usage: %s [-gl] [-effect] [-program] [-object] [-profile profile] [-entry entry] file\n", myProgramName);
}

int errorCount = 0;

void CgErrorHandler(CGcontext context, CGerror error, void *data)
{
    int* pCount = (int*)data;
    fprintf(stderr, "<> Error %i: %i %s\n", *pCount, error, cgGetErrorString(error));
    (*pCount)++;
    //<> exit(1);
}

static void InitContext(void)
{
    static int firstTime = 1;

    if (myCgContext) {
        return;
    }

    myCgContext = cgCreateContext();
    CheckForCgError("establishing Cg context");

    cgSetContextBehavior(myCgContext, CG_BEHAVIOR_CURRENT);
    CheckForCgError("setting context behavior");

    errorCount = 0;
    cgSetErrorHandler(CgErrorHandler, (void *)&errorCount);

    if (myStateType == STATE_TYPE_GL) {

        if (firstTime) {
            firstTime = 0;

            // minimal glut calls to create a GL context

            glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
            glutInitWindowSize(640, 480);
            glutCreateWindow("cgfxcat (OpenGL)");
        }

        cgGLRegisterStates(myCgContext);
        CheckForCgError("registering GL state handlers");

    } else {

        RegisterStates(myCgContext);
        CheckForCgError("registering state handlers");

        RegisterSamplerStates(myCgContext);
        CheckForCgError("registering sampler state handlers");
    }
}

static void DestroyContext(void)
{
    cgDestroyContext(myCgContext);
    myCgContext = 0;
    CheckForCgError("destroying Cg context");
}

static void CheckForCgError(const char *situation)
{
    CGerror error;
    CGerror firstError;
    const char *string = cgGetLastErrorString(&error);

    if (error != CG_NO_ERROR) {

        firstError = cgGetFirstError();

        tab_printf("Program: %s\n"
               "Situation: %s\n"
               "Error: %s\n"
               "First Error: %s\n",
               myProgramName, situation, string, cgGetErrorString(firstError));
        if (error == CG_COMPILER_ERROR) {
            tab_printf("\nCg compiler output...\n%s\n", cgGetLastListing(myCgContext));
        }
        exit(1);
    }
}

static int tab = 0;

static void tab_increment(void)
{
    ++tab;
}

static void tab_decrement(void)
{
    --tab;
    if (tab < 0) {
        tab = 0;
    }
}

static void tab_printf(char *format, ...)
{

#define BLANK_LINE_SIZE 512

    static char blank_line[BLANK_LINE_SIZE];
    static int firstTime = 1;

    int nSpaces;
    va_list ap;

    if (firstTime) {
        memset( blank_line, ' ', (size_t)BLANK_LINE_SIZE );
        firstTime = 0;
    }

    nSpaces = tab * 2;
    if (nSpaces >= BLANK_LINE_SIZE-1) {
        nSpaces = BLANK_LINE_SIZE-2;
    }

    blank_line[nSpaces+1] = '\0';

    va_start(ap, format);
    (void) printf("%s", blank_line);
    (void) vprintf(format, ap);
    va_end(ap);

    blank_line[nSpaces+1] = ' ';
}

static void tab_printf_enum(char* msg, CGenum e)
{
    tab_printf("%s: %s %i\n", msg, cgGetEnumString(e), cgGetEnum(cgGetEnumString(e)));
}

static void tab_printf_bool(char* msg, CGbool b)
{
    tab_printf("%s: %s\n", msg, (b) ? "CG_TRUE" : "CG_FALSE");
}

static void DumpSupportedProfiles(void)
{
    CGprofile profile;
    int nProfiles;
    int ii;

    // with the -gl flag we will output OGL profile options.
    // We have to have a GL context for that to work, so...

    if (myStateType == STATE_TYPE_GL) {
        InitContext();
    }

    nProfiles = cgGetNumSupportedProfiles();
    tab_printf("NumSupportedProfiles: %i\n", nProfiles);

    tab_increment();
    for (ii=0; ii<nProfiles; ++ii) {
        profile = cgGetSupportedProfile(ii);
        tab_printf("Profile %i: %s %i\n", ii, cgGetProfileString(profile),
                           cgGetProfile(cgGetProfileString(profile)));
        tab_increment();
        tab_printf_bool("IS_OPENGL_PROFILE",
            cgGetProfileProperty(profile, CG_IS_OPENGL_PROFILE));
        tab_printf_bool("IS_DIRECT3D_PROFILE",
            cgGetProfileProperty(profile, CG_IS_DIRECT3D_PROFILE));
        tab_printf_bool("IS_DIRECT3D_8_PROFILE",
            cgGetProfileProperty(profile, CG_IS_DIRECT3D_8_PROFILE));
        tab_printf_bool("IS_DIRECT3D_9_PROFILE",
            cgGetProfileProperty(profile, CG_IS_DIRECT3D_9_PROFILE));
        tab_printf_bool("IS_DIRECT3D_10_PROFILE",
            cgGetProfileProperty(profile, CG_IS_DIRECT3D_10_PROFILE));
        tab_printf_bool("IS_VERTEX_PROFILE",
            cgGetProfileProperty(profile, CG_IS_VERTEX_PROFILE));
        tab_printf_bool("IS_FRAGMENT_PROFILE",
            cgGetProfileProperty(profile, CG_IS_FRAGMENT_PROFILE));
        tab_printf_bool("IS_GEOMETRY_PROFILE",
            cgGetProfileProperty(profile, CG_IS_GEOMETRY_PROFILE));
        tab_printf_bool("IS_TRANSLATION_PROFILE",
            cgGetProfileProperty(profile, CG_IS_TRANSLATION_PROFILE));
        tab_printf_bool("IS_HLSL_PROFILE",
            cgGetProfileProperty(profile, CG_IS_HLSL_PROFILE));
        tab_printf_bool("IS_GLSL_PROFILE",
            cgGetProfileProperty(profile, CG_IS_GLSL_PROFILE));

        // dump the optimal options for OpenGL profiles

        if (myStateType == STATE_TYPE_GL &&
            cgGetProfileProperty(profile, CG_IS_OPENGL_PROFILE)) {
            DumpProfileOptions(profile);
        }

        tab_decrement();
    }
    tab_decrement();

    // now clean up the context

    if (myStateType == STATE_TYPE_GL) {
        DestroyContext();
    }
}

static void DumpProfileOptions(CGprofile profile)
{
    char const ** ppOptions;

    ppOptions = cgGLGetOptimalOptions(profile);

    if (ppOptions && *ppOptions) {

        tab_printf("Optimal options:\n");

        tab_increment();

        while (*ppOptions) {
            tab_printf("%s\n", *ppOptions);
            ppOptions++;
        }

        tab_decrement();
    }
}

static void DumpFile(const char * pFileName)
{
    CGeffect  effect;
    CGprogram program;
    CGenum    program_type = CG_SOURCE;

    tab_printf("File : %s\n", pFileName);

    tab_increment();

    InitContext();

    if (myFileType == CG_FILE_TYPE_EFFECT) {

        effect = cgCreateEffectFromFile(myCgContext, pFileName, NULL);
        CheckForCgError("creating effect");

        if (!effect) {
            tab_printf("%s\n", cgGetLastListing(myCgContext));
        } else {
            DumpContext(myCgContext);
            cgDestroyEffect(effect);
            CheckForCgError("destroying effect");
        }

    } else if (myFileType == CG_FILE_TYPE_PROGRAM_SOURCE ||
               myFileType == CG_FILE_TYPE_PROGRAM_OBJECT) {

        if (myFileType == CG_FILE_TYPE_PROGRAM_OBJECT) {
            GetProfileAndEntryFromObjectFile(pFileName);
            program_type = CG_OBJECT;
        }

        program = cgCreateProgramFromFile(myCgContext, program_type, pFileName,
                                          myProfile, myEntry, NULL);
        CheckForCgError("creating program");

        if (!program) {
            tab_printf("%s\n", cgGetLastListing(myCgContext));
        } else {
            DumpContext(myCgContext);
            cgDestroyProgram(program);
            CheckForCgError("destroying program");
        }
    }

    DestroyContext();

    tab_decrement();
}

static void DumpContext(CGcontext context)
{
    CGprogram program;
    CGeffect effect;
    CGstate state;
    int nProgram;
    int nEffect;
    int nState;
    CGbehavior b;

    tab_printf("Context: %i\n", PtrToUlong(context));

    tab_increment();

    tab_printf_bool("IsContext", cgIsContext(context));
    b = cgGetContextBehavior(context);
    tab_printf("Behavior: %s %i\n", cgGetBehaviorString(b), cgGetBehavior(cgGetBehaviorString(b)));
    tab_printf_enum("AutoCompile", cgGetAutoCompile(context));
    tab_printf_enum("ParameterSettingMode", cgGetParameterSettingMode(context));
    tab_printf("CompilerIncludeCallback: %p\n", cgGetCompilerIncludeCallback(context));
    tab_printf("LastListing: %s\n", cgGetLastListing(context));

    state = cgGetFirstState(context);
    if (state) {
        nState = 0;
        while (state) {
            DumpState(state, ++nState, 0);
            state = cgGetNextState(state);
        }
    }

    state = cgGetFirstSamplerState(context);
    if (state) {
        nState = 0;
        while (state) {
            DumpState(state, ++nState, 1);
            state = cgGetNextState(state);
        }
    }

    program = cgGetFirstProgram(context);

    nProgram = 0;
    while (program) {
        DumpProgram("Program", program, ++nProgram);
        program = cgGetNextProgram(program);
    }

    // dump effects

    effect = cgGetFirstEffect(context);

    nEffect = 0;
    while (effect) {
        DumpEffect(effect, ++nEffect);
        effect = cgGetNextEffect(effect);
    }

    tab_decrement();
}

static void DumpProgram(const char* pTitle, CGprogram program, int nProgram)
{
    CGprogram subProgram;
    CGprofile profile;
    CGdomain domain;
    CGannotation annotation;
    CGparameter parameter;
    CGbuffer buffer;
    CGtype userType;
    int nAnnotation;
    int nParameter;
    int nDomains;
    int nBuffers;
    int nUserTypes;
    int ii;
    char const * const * ppOptions;

    tab_printf("%s %i: %i %s %s\n", pTitle, nProgram, PtrToUlong(program),
                  cgGetProgramString(program, CG_PROGRAM_PROFILE),
                  cgGetProgramString(program, CG_PROGRAM_ENTRY));

    tab_increment();

    nDomains = cgGetNumProgramDomains(program);
    profile = cgGetProgramProfile(program);
    domain = cgGetProfileDomain(profile);

    if (nDomains > 1) {
        tab_increment();
        for (ii=0; ii<nDomains; ++ii) {
            tab_printf("DomainProfile: %s %i\n",
                    cgGetProfileString(cgGetProgramDomainProfile(program, ii)),
                    cgGetProgramDomainProfile(program, ii));
            subProgram = cgGetProgramDomainProgram(program, ii);
            DumpProgram("Subprogram", subProgram, ii);
        }
        tab_decrement();
        return;
    }

    tab_printf_bool("IsProgram", cgIsProgram(program));
    tab_printf("Context: %i\n", PtrToUlong(cgGetProgramContext(program)));
    tab_printf_bool("IsProgramCompiled", cgIsProgramCompiled(program));

    if (!cgIsProgramCompiled(program)) {
        cgCompileProgram(program);
    }

    tab_printf("Profile: %s %i\n", cgGetProfileString(profile),
                           cgGetProfile(cgGetProfileString(profile)));
    tab_printf("Domain: %s\n", cgGetDomainString(domain));
    tab_printf_bool("IsProfileSupported", cgIsProfileSupported(profile));
    tab_printf_enum("Input", cgGetProgramInput(program));
    tab_printf_enum("Output", cgGetProgramOutput(program));

    ppOptions = cgGetProgramOptions(program);
    tab_printf("Options:");
    while (*ppOptions) {
        printf(" %s", *ppOptions);
        ++ppOptions;
    }
    printf("\n");

    tab_printf("Source:\n%s\n", cgGetProgramString(program, CG_PROGRAM_SOURCE));
    tab_printf("Object:\n%s\n", cgGetProgramString(program, CG_COMPILED_PROGRAM));

    tab_printf("BufferMaxSize: %i\n", cgGetProgramBufferMaxSize(profile));
    nBuffers = cgGetProgramBufferMaxIndex(profile);
    tab_printf("BufferIndex: Buffer BufferSize\n");
    tab_increment();
    for (ii=0; ii<nBuffers; ++ii) {
        buffer = cgGetProgramBuffer(program, ii);
        tab_printf("%i: %i %i\n", ii+1, PtrToUlong(buffer),
            (buffer) ? cgGetBufferSize(buffer) : 0);
    }
    tab_decrement();

    nUserTypes = cgGetNumUserTypes(program);
    tab_printf("NumUserTypes: %i\n", nUserTypes);

    if (nUserTypes > 0) {
        for (ii=0; ii<nUserTypes; ++ii) {
            userType = cgGetUserType(program, ii);
            DumpType("UserType", ii+1, userType);
        }
    }

    annotation = cgGetFirstProgramAnnotation(program);

    nAnnotation = 0;
    while (annotation) {
        DumpAnnotation(annotation, ++nAnnotation, program, 4);
        annotation = cgGetNextAnnotation(annotation);
    }

    parameter = cgGetFirstParameter(program, CG_GLOBAL);
    nParameter = 0;
    while (parameter) {
        DumpParameter("Global Parameter", parameter, ++nParameter, program, 0);
        parameter = cgGetNextParameter(parameter);
    }

    parameter = cgGetFirstParameter(program, CG_PROGRAM);
    nParameter = 0;
    while (parameter) {
        DumpParameter("Program Parameter", parameter, ++nParameter, program, 0);
        parameter = cgGetNextParameter(parameter);
    }

    parameter = cgGetFirstLeafParameter(program, CG_GLOBAL);
    nParameter = 0;
    while (parameter) {
        DumpParameter("Global Leaf Parameter", parameter, ++nParameter, program, 0);
        parameter = cgGetNextLeafParameter(parameter);
    }

    parameter = cgGetFirstLeafParameter(program, CG_PROGRAM);
    nParameter = 0;
    while (parameter) {
        DumpParameter("Program Leaf Parameter", parameter, ++nParameter, program, 0);
        parameter = cgGetNextLeafParameter(parameter);
    }

    tab_decrement();
}

static void DumpEffect(CGeffect effect, int nEffect)
{
    CGcontext context;
    CGparameter parameter;
    CGtechnique technique;
    const char * pName;
    CGannotation annotation;
    CGtype userType;
    int nAnnotation;
    int nParameter;
    int nTechnique;
    int nUserTypes;
    int ii;

    pName = cgGetEffectName(effect);

    tab_printf("Effect %i: %i name=\"%s\"\n", nEffect, PtrToUlong(effect), pName);

    tab_increment();

    tab_printf_bool("IsEffect", cgIsEffect(effect));
    context = cgGetEffectContext(effect);
    tab_printf("Context: %i\n", PtrToUlong(context));

    if (pName) {
        tab_printf("NamedEffect: %i\n", PtrToUlong(cgGetNamedEffect(context, pName)));
    }

    nUserTypes = cgGetNumUserTypes(effect);
    tab_printf("NumUserTypes: %i\n", nUserTypes);

    if (nUserTypes > 0) {
        for (ii=0; ii<nUserTypes; ++ii) {
            userType = cgGetUserType(effect, ii);
            DumpType("UserType", ii+1, userType);
        }
    }

    annotation = cgGetFirstEffectAnnotation(effect);

    nAnnotation = 0;
    while (annotation) {
        DumpAnnotation(annotation, ++nAnnotation, effect, 1);
        annotation = cgGetNextAnnotation(annotation);
    }

    parameter = cgGetFirstEffectParameter(effect);
    nParameter = 0;
    while (parameter) {
        DumpParameter("Parameter", parameter, ++nParameter, effect, 0);
        parameter = cgGetNextParameter(parameter);
    }

    parameter = cgGetFirstLeafEffectParameter(effect);
    nParameter = 0;
    while (parameter) {
        DumpParameter("Leaf Parameter", parameter, ++nParameter, effect, 0);
        parameter = cgGetNextLeafParameter(parameter);
    }

    technique = cgGetFirstTechnique(effect);
    nTechnique = 0;
    while (technique) {
        DumpTechnique(technique, ++nTechnique);
        technique = cgGetNextTechnique(technique);
    }

    tab_decrement();
}

static void DumpTechnique(CGtechnique technique, int nTechnique)
{
    CGeffect effect;
    CGpass pass;
    CGparameter parameter1;
    CGparameter parameter2;
    CGannotation annotation;
    const char * pName;
    int nPass;
    int nParameter;
    int nAnnotation;

    pName = cgGetTechniqueName(technique);

    tab_printf("Technique %i: %i name=\"%s\"\n", nTechnique, PtrToUlong(technique), pName);

    tab_increment();

    tab_printf_bool("IsTechnique", cgIsTechnique(technique));
    tab_printf_bool("IsValidated", cgIsTechniqueValidated(technique));

    if (!cgIsTechniqueValidated(technique)) {
        if (cgValidateTechnique(technique) == CG_FALSE) {
            tab_printf("Technique did not validate.\n");
            tab_printf("%s\n", cgGetLastListing(myCgContext));
        }
    }

    effect = cgGetTechniqueEffect(technique);
    tab_printf("Effect: %i\n", PtrToUlong(effect));
    if (pName && effect) {
        tab_printf("NamedTechnique: %i\n", PtrToUlong(cgGetNamedTechnique(effect, pName)));
    }

    parameter1 = (effect) ? cgGetFirstEffectParameter(effect) : 0;
    parameter2 = (effect) ? cgGetFirstLeafEffectParameter(effect) : 0;

    if (parameter1 || parameter2) {
        tab_printf("IsParameterUsed:\n");
        tab_increment();
        nParameter = 0;
        while (parameter1) {
            tab_printf("Effect Parameter %i: %i %s %s\n", ++nParameter,
                parameter1, cgGetParameterName(parameter1),
                cgIsParameterUsed(parameter1, technique) ? "CG_TRUE" : "CG_FALSE");
            parameter1 = cgGetNextParameter(parameter1);
        }
        nParameter = 0;
        while (parameter2) {
            tab_printf("Leaf Effect Parameter %i: %i %s %s\n", ++nParameter,
                parameter2, cgGetParameterName(parameter2),
                cgIsParameterUsed(parameter2, technique) ? "CG_TRUE" : "CG_FALSE");
            parameter2 = cgGetNextLeafParameter(parameter2);
        }
        tab_decrement();
    }

    annotation = cgGetFirstTechniqueAnnotation(technique);

    nAnnotation = 0;
    while (annotation) {
        DumpAnnotation(annotation, ++nAnnotation, technique, 5);
        annotation = cgGetNextAnnotation(annotation);
    }

    pass = cgGetFirstPass(technique);

    nPass = 0;
    while (pass) {
        DumpPass(pass, ++nPass);
        pass = cgGetNextPass(pass);
    }

    tab_decrement();
}

static void DumpPass(CGpass pass, int nPass)
{
    const char * pName;
    CGstateassignment sa;
    CGannotation annotation;
    CGparameter parameter1;
    CGparameter parameter2;
    CGeffect effect;
    CGtechnique technique;
    int nStateAssignment;
    int nAnnotation;
    int nParameter;

    pName = cgGetPassName(pass);

    tab_printf("Pass %i: %i name=\"%s\"\n", nPass, PtrToUlong(pass), pName);

    tab_increment();

    tab_printf_bool("IsPass", cgIsPass(pass));

    technique = cgGetPassTechnique(pass);
    tab_printf("Technique: %i\n", PtrToUlong(technique));

    if (pName && technique) {
        tab_printf("NamedPass: %i\n", PtrToUlong(cgGetNamedPass(technique, pName)));
    }

    effect = (technique) ? cgGetTechniqueEffect(technique) : 0;
    tab_printf("Effect: %i\n", PtrToUlong(effect));

    parameter1 = (effect) ? cgGetFirstEffectParameter(effect) : 0;
    parameter2 = (effect) ? cgGetFirstLeafEffectParameter(effect) : 0;

    if (parameter1 || parameter2) {
        tab_printf("IsParameterUsed:\n");
        tab_increment();
        nParameter = 0;
        while (parameter1) {
            tab_printf("Effect Parameter %i: %i %s %s\n", ++nParameter,
                parameter1, cgGetParameterName(parameter1),
                cgIsParameterUsed(parameter1, pass) ? "CG_TRUE" : "CG_FALSE");
            parameter1 = cgGetNextParameter(parameter1);
        }
        nParameter = 0;
        while (parameter2) {
            tab_printf("Leaf Effect Parameter %i: %i %s %s\n", ++nParameter,
                parameter2, cgGetParameterName(parameter2),
                cgIsParameterUsed(parameter2, pass) ? "CG_TRUE" : "CG_FALSE");
            parameter2 = cgGetNextLeafParameter(parameter2);
        }
        tab_decrement();
    }

    tab_printf("TessellationControlProgram: %i\n", PtrToUlong(cgGetPassProgram(pass,CG_TESSELLATION_CONTROL_DOMAIN)));
    tab_printf("TessellationEvaluationProgram: %i\n", PtrToUlong(cgGetPassProgram(pass,CG_TESSELLATION_EVALUATION_DOMAIN)));
    tab_printf("VertexProgram: %i\n", PtrToUlong(cgGetPassProgram(pass,CG_VERTEX_DOMAIN)));
    tab_printf("FragmentProgram: %i\n", PtrToUlong(cgGetPassProgram(pass,CG_FRAGMENT_DOMAIN)));
    tab_printf("GeometryProgram: %i\n", PtrToUlong(cgGetPassProgram(pass,CG_GEOMETRY_DOMAIN)));

    annotation = cgGetFirstPassAnnotation(pass);

    nAnnotation = 0;
    while (annotation) {
        DumpAnnotation(annotation, ++nAnnotation, pass, 3);
        annotation = cgGetNextAnnotation(annotation);
    }

    sa = cgGetFirstStateAssignment(pass);

    nStateAssignment = 0;
    while (sa) {
        DumpStateAssignment(sa, ++nStateAssignment, 0);
        sa = cgGetNextStateAssignment(sa);
    }

    tab_decrement();
}

static void DumpStateAssignment(CGstateassignment sa, int nStateAssignment, int isSSA)
{
    const char * pName;
    CGstate state;
    CGtype type;
    CGpass pass;
    CGtechnique technique;
    CGeffect effect;
    CGparameter parameter;
    CGparameter parameter1;
    CGparameter parameter2;
    CGprogram program;
    int index;
    int nDependents;
    int nParameter;
    int nValues;
    int ii;
    const float *fvalues;
    const int *ivalues;
    const CGbool *bvalues;

    index = cgGetStateAssignmentIndex(sa);
    pass = cgGetStateAssignmentPass(sa);

    if (isSSA) {
        state = cgGetSamplerStateAssignmentState(sa);
    } else {
        state = cgGetStateAssignmentState(sa);
    }

    pName = cgGetStateName(state);
    type = cgGetStateType(state);

    if (isSSA) {
        tab_printf("SamplerStateAssignment %i: %i\n", nStateAssignment, PtrToUlong(sa));
    } else {
        tab_printf("StateAssignment %i: %i\n", nStateAssignment, PtrToUlong(sa));
    }

    tab_increment();

    tab_printf_bool("IsStateAssigment", cgIsStateAssignment(sa));
    tab_printf("State Name: %s\n", pName);
    DumpType("State Type", -1, type);

    if (isSSA) {
        parameter = cgGetSamplerStateAssignmentParameter(sa);
        if (parameter) {
            tab_printf("SamplerStateAssignmentParameter: %i %s\n",
                            PtrToUlong(parameter), cgGetParameterName(parameter));
        }
    }

    if (pName) {
        if (pass) {
            tab_printf("NamedStateAssignment: %i\n",
               PtrToUlong(cgGetNamedStateAssignment(pass, pName)));
        }
        if (isSSA) {
            parameter = cgGetSamplerStateAssignmentParameter(sa);
            if (parameter) {
                tab_printf("NamedSamplerStateAssignment: %i\n",
                                cgGetNamedSamplerStateAssignment(parameter, pName));
            }
        }
    }

    technique = (pass) ? cgGetPassTechnique(pass) : 0;
    effect = (technique) ? cgGetTechniqueEffect(technique) : 0;

    parameter1 = (effect) ? cgGetFirstEffectParameter(effect) : 0;
    parameter2 = (effect) ? cgGetFirstLeafEffectParameter(effect) : 0;

    if (parameter1 || parameter2) {
        tab_printf("IsParameterUsed:\n");
        tab_increment();
        nParameter = 0;
        while (parameter1) {
            tab_printf("Effect Parameter %i: %i %s %s\n", ++nParameter,
                parameter1, cgGetParameterName(parameter1),
                cgIsParameterUsed(parameter1, sa) ? "CG_TRUE" : "CG_FALSE");
            parameter1 = cgGetNextParameter(parameter1);
        }
        nParameter = 0;
        while (parameter2) {
            tab_printf("Leaf Effect Parameter %i: %i %s %s\n", ++nParameter,
                parameter2, cgGetParameterName(parameter2),
                cgIsParameterUsed(parameter2, sa) ? "CG_TRUE" : "CG_FALSE");
            parameter2 = cgGetNextLeafParameter(parameter2);
        }
        tab_decrement();
    }

    parameter = cgGetConnectedStateAssignmentParameter(sa);
    if (parameter) {
        tab_printf("ConnectedStateAssignmentParameter: %i %s\n",
                        PtrToUlong(parameter), cgGetParameterName(parameter));
    }

    switch (type) {
    case CG_PROGRAM_TYPE:
        program = cgGetProgramStateAssignmentValue(sa);
        tab_printf("Value: %i %s %s\n", PtrToUlong(program),
                  cgGetProgramString(program, CG_PROGRAM_PROFILE),
                  cgGetProgramString(program, CG_PROGRAM_ENTRY));
        break;
    case CG_FLOAT:
    case CG_FLOAT1:
    case CG_FLOAT2:
    case CG_FLOAT3:
    case CG_FLOAT4:
        fvalues = cgGetFloatStateAssignmentValues(sa, &nValues);
        tab_printf("Value(s):");
        for (ii=0; ii<nValues; ++ii) {
            printf(" %f", fvalues[ii]);
        }
        printf("\n");
        break;
    case CG_INT:
    case CG_INT1:
    case CG_INT2:
    case CG_INT3:
    case CG_INT4:
        ivalues = cgGetIntStateAssignmentValues(sa, &nValues);
        tab_printf("Value(s):");
        for (ii=0; ii<nValues; ++ii) {
            printf(" %d (0x%x)", ivalues[ii], ivalues[ii]);
        }
        printf("\n");
        break;
    case CG_BOOL:
    case CG_BOOL1:
    case CG_BOOL2:
    case CG_BOOL3:
    case CG_BOOL4:
        bvalues = cgGetBoolStateAssignmentValues(sa, &nValues);
        tab_printf("Value(s):");
        for (ii=0; ii<nValues; ++ii) {
            printf(" %s", bvalues[ii] ? "true" : "false");
        }
        printf("\n");
        break;
    case CG_STRING:
        tab_printf("Value: %s\n", cgGetStringStateAssignmentValue(sa));
        break;
    case CG_TEXTURE:
        tab_printf("Value: 0x%p\n", cgGetTextureStateAssignmentValue(sa));
        break;
    case CG_SAMPLER1D:
    case CG_SAMPLER2D:
    case CG_SAMPLER3D:
    case CG_SAMPLERCUBE:
    case CG_SAMPLERRECT:
        tab_printf("Value: 0x%p\n", cgGetSamplerStateAssignmentValue(sa));
        break;
    default:
        printf("UNEXPECTED State Assignment Type: %s 0x%x (%d)\n",
                  cgGetTypeString(type), type, type);
        break;
    }

    tab_printf("Pass: %i\n", PtrToUlong(pass));
    tab_printf("Index: %i\n", index);

    nDependents = cgGetNumDependentStateAssignmentParameters(sa);

    if (nDependents > 0) {
        tab_printf("DependentStateAssignmentParameters:\n");
        tab_increment();
        for (ii=0; ii<nDependents; ++ii) {
            parameter = cgGetDependentStateAssignmentParameter(sa, ii);
            tab_printf("%i: %i %s\n", ii+1, PtrToUlong(parameter),
                                cgGetParameterName(parameter));
        }
        tab_decrement();
    }

    nDependents = cgGetNumDependentProgramArrayStateAssignmentParameters(sa);

    if (nDependents > 0) {
        tab_printf("DependentProgramArrayStateAssignmentParameters:\n");
        tab_increment();
        for (ii=0; ii<nDependents; ++ii) {
            parameter = cgGetDependentProgramArrayStateAssignmentParameter(sa, ii);
            tab_printf("%i: %i %s\n", ii+1, PtrToUlong(parameter),
                                cgGetParameterName(parameter));
        }
        tab_decrement();
    }

    tab_decrement();
}

static void DumpAnnotation(CGannotation annotation, int nAnnotation,
                           CGhandle container, int containerType)
{
    const char * pName;
    CGtype type;
    CGparameter parameter;
    const float *fvalues;
    const int *ivalues;
    const CGbool *bvalues;
    const char * const * svalues;
    int nValues;
    int nDependents;
    int ii;

    pName = cgGetAnnotationName(annotation);

    tab_printf("Annotation %i: %i name=\"%s\"\n", nAnnotation, PtrToUlong(annotation), pName);

    tab_increment();

    type = cgGetAnnotationType(annotation);

    tab_printf_bool("IsAnnotation", cgIsAnnotation(annotation));

    if (pName && container) {
        if (containerType == 1) {
            tab_printf("NamedEffectAnnotation: %i\n",
              PtrToUlong(cgGetNamedEffectAnnotation(container, pName)));
        } else if (containerType == 2) {
            tab_printf("cgGetNamedParameterAnnotation: %i\n",
              PtrToUlong(cgGetNamedParameterAnnotation(container, pName)));
        } else if (containerType == 3) {
            tab_printf("cgGetNamedPassAnnotation: %i\n",
              PtrToUlong(cgGetNamedPassAnnotation(container, pName)));
        } else if (containerType == 4) {
            tab_printf("cgGetNamedProgramAnnotation: %i\n",
              PtrToUlong(cgGetNamedProgramAnnotation(container, pName)));
        } else if (containerType == 5) {
            tab_printf("cgGetNamedTechniqueAnnotation: %i\n",
              PtrToUlong(cgGetNamedTechniqueAnnotation(container, pName)));
        }
    }

    DumpType("Type", -1, type);

    switch (type) {
    case CG_FLOAT:
    case CG_FLOAT1:
    case CG_FLOAT2:
    case CG_FLOAT3:
    case CG_FLOAT4:
        fvalues = cgGetFloatAnnotationValues(annotation, &nValues);
        tab_printf("Value(s):");
        for (ii=0; ii<nValues; ++ii) {
            printf(" %f", fvalues[ii]);
        }
        printf("\n");
        break;
    case CG_INT:
    case CG_INT1:
    case CG_INT2:
    case CG_INT3:
    case CG_INT4:
        ivalues = cgGetIntAnnotationValues(annotation, &nValues);
        tab_printf("Value(s):");
        for (ii=0; ii<nValues; ++ii) {
            printf(" %d (0x%x)", ivalues[ii], ivalues[ii]);
        }
        printf("\n");
        break;
    case CG_BOOL:
    case CG_BOOL1:
    case CG_BOOL2:
    case CG_BOOL3:
    case CG_BOOL4:
        bvalues = cgGetBoolAnnotationValues(annotation, &nValues);
        tab_printf("Value(s):");
        for (ii=0; ii<nValues; ++ii) {
            printf(" %s", bvalues[ii] ? "true" : "false");
        }
        printf("\n");
        break;
    case CG_STRING:
        tab_printf("Value: %s\n", cgGetStringAnnotationValue(annotation));
        svalues = cgGetStringAnnotationValues(annotation, &nValues);
        tab_printf("Value(s):");
        for (ii=0; ii<nValues; ++ii) {
            printf(" \"%s\"", svalues[ii]);
        }
        printf("\n");
        break;
        break;
    case CG_TEXTURE:
    case CG_SAMPLER1D:
    case CG_SAMPLER2D:
    case CG_SAMPLER3D:
    case CG_SAMPLERCUBE:
    case CG_SAMPLERRECT:
    case CG_PROGRAM_TYPE:
    default:
        printf("UNEXPECTED Annotation Type: %s 0x%x (%d)\n",
                  cgGetTypeString(type), type, type);
        break;
    }

    nDependents = cgGetNumDependentAnnotationParameters(annotation);

    if (nDependents > 0) {
        tab_printf("DependentAnnotationParameters:\n");
        tab_increment();
        for (ii=0; ii<nDependents; ++ii) {
            parameter = cgGetDependentAnnotationParameter(annotation, ii);
            tab_printf("%i: %i %s\n", ii+1, PtrToUlong(parameter),
                                cgGetParameterName(parameter));
        }
        tab_decrement();
    }

    tab_decrement();
}

static void DumpState(CGstate state, int nState, int isSamplerState)
{
    CGcontext context;
    CGtype type;
    CGprofile latestProfile;
    const char* pLatestProfileString;
    const char* pName;
    int ii;
    int nEnums;
    int enumValue;

    pName = cgGetStateName(state);

    if (isSamplerState) {
        tab_printf("SamplerState %i: %i %s\n", nState, PtrToUlong(state), pName);
    } else {
        tab_printf("State %i: %i %s\n", nState, PtrToUlong(state), pName);
    }

    tab_increment();

    type = cgGetStateType(state);
    DumpType("Type", -1, type);

    latestProfile = cgGetStateLatestProfile(state);
    if (latestProfile == CG_PROFILE_UNKNOWN) {
        pLatestProfileString = "unknown";
    } else {
        pLatestProfileString = cgGetProfileString(latestProfile);
    }

    tab_printf_bool("IsState", cgIsState(state));

    context = cgGetStateContext(state);
    tab_printf("Context: %i\n", PtrToUlong(context));

    if (pName && context) {
        if (isSamplerState) {
            tab_printf("NamedSamplerState: %i\n",
                            PtrToUlong(cgGetNamedSamplerState(context, pName)));
        } else {
            tab_printf("NamedState: %i\n", PtrToUlong(cgGetNamedState(context, pName)));
        }
    }

    tab_printf("LatestProfile: %s %i\n", pLatestProfileString, (int)latestProfile);
    tab_printf("SetCallback: %p\n", cgGetStateSetCallback(state));
    tab_printf("ResetCallback: %p\n", cgGetStateResetCallback(state));
    tab_printf("ValidateCallback: %p\n", cgGetStateValidateCallback(state));

    nEnums = cgGetNumStateEnumerants(state);
    tab_printf("NumStateEnumerants: %i\n", nEnums);

    if (nEnums > 0) {
        tab_printf("Enumerants:\n");
        tab_increment();
        for (ii=0; ii<nEnums; ++ii) {
            pName = cgGetStateEnumerant(state, ii, &enumValue );
            tab_printf("%i: %s %i\n", ii+1, pName, enumValue);
        }
        tab_decrement();
    }

    tab_decrement();
}

static void DumpType(const char* pTitle, int index, CGtype type)
{
    CGtype base;
    CGparameterclass parameterclass;
    CGbool isMatrix;
    CGtype parentType;
    int ii;
    int nRows;
    int nCols;
    int nParentTypes;

    if (index < 0) {
        tab_printf("%s: %s %i\n", pTitle,
              cgGetTypeString(type), cgGetType(cgGetTypeString(type)));
    } else {
        tab_printf("%s %i: %s %i\n", pTitle, index,
              cgGetTypeString(type), cgGetType(cgGetTypeString(type)));
    }

    tab_increment();

    base = cgGetTypeBase(type);
    parameterclass = cgGetTypeClass(type);

    tab_printf_bool("IsInterfaceType", cgIsInterfaceType(type));
    tab_printf("Base: %s %i\n", cgGetTypeString(base), cgGetType(cgGetTypeString(base)));
    tab_printf("ParameterClass: %s %i\n",
                 cgGetParameterClassString(parameterclass), parameterclass);

    isMatrix = cgGetTypeSizes(type, &nRows, &nCols);
    tab_printf_bool("IsMatrix", isMatrix);
    tab_printf("TypeSizes ncols: %i\n", nCols);
    tab_printf("TypeSizes nrows: %i\n", nRows);

    cgGetMatrixSize(type, &nRows, &nCols);
    tab_printf("MatrixSize ncols: %i\n", nCols);
    tab_printf("MatrixSize nrows: %i\n", nRows);

    nParentTypes = cgGetNumParentTypes(type);
    tab_printf("NumParentTypes: %i\n", nParentTypes);

    if (nParentTypes > 0) {
        tab_printf("ParentTypes:\n");
        tab_increment();
        for (ii=0; ii<nParentTypes; ++ii) {
            parentType = cgGetParentType(type, ii);
            tab_printf("%i: %i %s IsParentType=%s\n",
               ii+1, (int)parentType, cgGetTypeString(parentType),
               (cgIsParentType(parentType,type)) ? "CG_TRUE" : "CG_FALSE");
        }
        tab_decrement();
    }

    tab_decrement();
}

static void DumpParameter(const char* pTitle, CGparameter parameter, int nParameter,
                          CGhandle container, CGparameter parent)
{
    const char * pName;
    CGannotation annotation;
    CGparameter source;
    CGparameter connected;
    CGparameter structParameter;
    CGparameter arrayParameter;
    CGeffect effect;
    CGprogram program;
    CGstateassignment sa;
    CGtype type;
    const char* pSemantic;
    int ii;
    int nConnected;
    int nAnnotation;
    int nDimension;
    int nStateAssignment;
    int size;

    pName = cgGetParameterName(parameter);

    tab_printf("%s %i: %i %s\n", pTitle, nParameter, PtrToUlong(parameter), pName);

    tab_increment();

    type = cgGetParameterType(parameter);

    tab_printf_bool("IsParameter", cgIsParameter(parameter));
    tab_printf_bool("IsParameterGlobal", cgIsParameterGlobal(parameter));
    tab_printf_bool("IsParameterReferenced", cgIsParameterReferenced(parameter));
    tab_printf("IsParameterUsed: %s container=%i\n",
        cgIsParameterUsed(parameter, container) ? "CG_TRUE" : "CG_FALSE", PtrToUlong(container));

    tab_printf("Context: %i\n", cgGetParameterContext(parameter));
    program = cgGetParameterProgram(parameter);
    tab_printf("Program: %i\n", program);
    effect = cgGetParameterEffect(parameter);
    tab_printf("Effect: %i\n", effect);
    tab_printf_enum("Variability", cgGetParameterVariability(parameter));
    tab_printf_enum("Direction", cgGetParameterDirection(parameter));
    pSemantic = cgGetParameterSemantic(parameter);
    tab_printf("Semantic: %s\n", pSemantic);

    if (pName) {
        if (program) {
            tab_printf("NamedParameter: %i\n",
                  cgGetNamedParameter(program, pName));
            tab_printf("GLOBAL NamedProgramParameter: %i\n",
                  cgGetNamedProgramParameter(program, CG_GLOBAL, pName));
            tab_printf("PROGRAM NamedProgramParameter: %i\n",
                  cgGetNamedProgramParameter(program, CG_PROGRAM, pName));
        }

        if (effect) {
            tab_printf("NamedEffectParameter: %i\n",
                  cgGetNamedEffectParameter(effect, pName));
        }
        if (parent) {
            tab_printf("NamedSubParameter: %i\n",
                  cgGetNamedSubParameter(parameter, pName));
            if (cgGetParameterType(parent) == CG_STRUCT) {
                tab_printf("NamedStructParameter: %i\n",
                      cgGetNamedStructParameter(parameter, pName));
            }
        }
    }

    if (effect && pSemantic && pSemantic[0] != '\0') {
        tab_printf("EffectParameterBySemantic: %i\n",
              cgGetEffectParameterBySemantic(effect, pSemantic));
    }

    tab_printf("OrdinalNumber: %i\n", cgGetParameterOrdinalNumber(parameter));
    DumpType("Type", -1, type);
    DumpType("NamedType", -1, cgGetParameterNamedType(parameter));
    DumpType("BaseType", -1, cgGetParameterBaseType(parameter));
    DumpType("ResourceType", -1, cgGetParameterResourceType(parameter));
    tab_printf("ResourceSize: %i\n", cgGetParameterResourceSize(parameter));
    tab_printf("BufferIndex: %i\n", cgGetParameterBufferIndex(parameter));
    tab_printf("BufferOffset: %i\n", cgGetParameterBufferOffset(parameter));

    if (type == CG_ARRAY) {
        tab_printf("ArrayIndex: %i\n", cgGetParameterIndex(parameter));
        DumpType("ArrayType", -1, cgGetArrayType(parameter));
        nDimension = cgGetArrayDimension(parameter);
        tab_printf("ArrayDimension: %i\n", nDimension);
        tab_printf("ArrayTotalSize: %i\n", cgGetArrayTotalSize(parameter));
        tab_printf("ArraySize:\n");
        tab_increment();
        for (ii=0; ii<nDimension; ++ii) {
            tab_printf("%i: %i\n", ii, cgGetArraySize(parameter, ii));
        }
        tab_decrement();
    }

    annotation = cgGetFirstParameterAnnotation(parameter);

    nAnnotation = 0;
    while (annotation) {
        DumpAnnotation(annotation, ++nAnnotation, parameter, 2);
        annotation = cgGetNextAnnotation(annotation);
    }

    sa = cgGetFirstSamplerStateAssignment(parameter);
    nStateAssignment = 0;
    while (sa) {
        DumpStateAssignment(sa, ++nStateAssignment, 1);
        sa = cgGetNextStateAssignment(sa);
    }

    source = cgGetConnectedParameter(parameter);
    if (source) {
        tab_printf("SourceParameter: %i %s\n", PtrToUlong(source), cgGetParameterName(source));
    }

    nConnected = cgGetNumConnectedToParameters(parameter);

    if (nConnected > 0) {
        tab_printf("ConnectedParameters:\n");
        tab_increment();
        for (ii=0; ii<nConnected; ++ii) {
            connected = cgGetConnectedToParameter(parameter, ii);
            tab_printf("%i: %i %s\n", ii+1, PtrToUlong(connected),
                                cgGetParameterName(connected));
        }
        tab_decrement();
    }

    if (type == CG_ARRAY) {
        size = cgGetArraySize(parameter, 0);
        for (ii=0; ii<size; ++ii) {
            arrayParameter = cgGetArrayParameter(parameter, ii);
            DumpParameter("ArrayParameter", arrayParameter, ii+1, container, parent);
        }
    } else if (type == CG_STRUCT) {
        structParameter = cgGetFirstStructParameter(parameter);
        nParameter = 0;
        while (structParameter) {
            DumpParameter("StructParameter", structParameter, ++nParameter,
                          container, parent);
            structParameter = cgGetNextParameter(structParameter);
        }
    } else {
        tab_printf("Resource: %s %i\n",
             cgGetResourceString(cgGetParameterResource(parameter)),
             cgGetResource(cgGetResourceString(cgGetParameterResource(parameter))));
        tab_printf("BaseResource: %s\n", cgGetResourceString(cgGetParameterBaseResource(parameter)));
        tab_printf("ResourceName: %s\n", cgGetParameterResourceName(parameter));
        tab_printf("ResourceIndex: %i\n", cgGetParameterResourceIndex(parameter));

        source = cgGetFirstDependentParameter(parameter);

        if (source) {
            tab_printf("DependentParameters:\n");
            tab_increment();
            ii = 0;
            while (source) {
                tab_printf("%i: %i %s\n", ++ii, PtrToUlong(source),
                    cgGetParameterName(source));
                source = cgGetNextParameter(source);
            }
            tab_decrement();
        }

        DumpParameterValues(parameter);
    }

    tab_decrement();
}

static void DumpParameterValues(CGparameter parameter)
{

    float fvalues[16];
    int ivalues[16];
    int ii;
    int jj;
    int nValues;
    CGtype type;
    CGtype basetype;
    CGparameterclass parameterclass;
    int nRows;
    int nCols;

    type = cgGetParameterType(parameter);

    if ((type == CG_ARRAY) || (type == CG_STRUCT)) {
        return;
    }

    basetype = cgGetParameterBaseType(parameter);
    parameterclass = cgGetParameterClass(parameter);

    nRows = cgGetParameterRows(parameter);
    nCols = cgGetParameterColumns(parameter);

    tab_printf("Columns: %i\n", nCols);
    tab_printf("Rows: %i\n", nRows);
    tab_printf("ArrayTotalSize: %i\n", cgGetArrayTotalSize(parameter));
    tab_printf("ParameterClass: %s %i\n",
                 cgGetParameterClassString(parameterclass), parameterclass);

    switch (parameterclass) {
    case CG_PARAMETERCLASS_STRUCT:
    case CG_PARAMETERCLASS_ARRAY:
        // should have been caught above
        return;
    case CG_PARAMETERCLASS_SCALAR:
    case CG_PARAMETERCLASS_VECTOR:
        switch (basetype) {
        case CG_FLOAT:
        case CG_HALF:
        case CG_FIXED:
            nValues = cgGetParameterDefaultValuefr(parameter, 16, fvalues);
            tab_printf("DefaultValue(s):");
            for (ii=0; ii<nValues; ++ii) {
                printf(" %f", fvalues[ii]);
            }
            printf("\n");
            nValues = cgGetParameterValuefr(parameter, 16, fvalues);
            tab_printf("Value(s):");
            for (ii=0; ii<nValues; ++ii) {
                printf(" %f", fvalues[ii]);
            }
            printf("\n");
            break;
        case CG_INT:
        case CG_BOOL:
            nValues = cgGetParameterDefaultValueir(parameter, 16, ivalues);
            tab_printf("DefaultValue(s):");
            for (ii=0; ii<nValues; ++ii) {
                printf(" %i", ivalues[ii]);
            }
            printf("\n");
            nValues = cgGetParameterValueir(parameter, 16, ivalues);
            tab_printf("Value(s):");
            for (ii=0; ii<nValues; ++ii) {
                printf(" %i", ivalues[ii]);
            }
            printf("\n");
            break;
        default:
            printf("UNEXPECTED Parameter BaseType: %s 0x%x (%d)\n",
                      cgGetTypeString(basetype), basetype, basetype);
            break;
        }
        break;
    case CG_PARAMETERCLASS_MATRIX:
        switch (basetype) {
        case CG_FLOAT:
        case CG_HALF:
        case CG_FIXED:
            nValues = cgGetParameterDefaultValuefr(parameter, 16, fvalues);
            tab_printf("DefaultValue(s):\n");
            tab_increment();
            for (ii=0; ii<nRows; ++ii) {
                tab_printf("Row %i:", ii+1);
                for (jj=0; jj<nCols; ++jj) {
                    printf(" %f", fvalues[ii*nCols+jj]);
                }
                printf("\n");
            }
            tab_decrement();
            nValues = cgGetParameterValuefr(parameter, 16, fvalues);
            tab_printf("Value(s):\n");
            tab_increment();
            for (ii=0; ii<nRows; ++ii) {
                tab_printf("Row %i:", ii+1);
                for (jj=0; jj<nCols; ++jj) {
                    printf(" %f", fvalues[ii*nCols+jj]);
                }
                printf("\n");
            }
            tab_decrement();
            break;
        case CG_INT:
        case CG_BOOL:
            nValues = cgGetParameterDefaultValueir(parameter, 16, ivalues);
            tab_printf("DefaultValue(s):\n");
            tab_increment();
            for (ii=0; ii<nRows; ++ii) {
                tab_printf("Row %i:", ii+1);
                for (jj=0; jj<nCols; ++jj) {
                    printf(" %i", ivalues[ii*nCols+jj]);
                }
                printf("\n");
            }
            tab_decrement();
            nValues = cgGetParameterValueir(parameter, 16, ivalues);
            tab_printf("Value(s):\n");
            tab_increment();
            for (ii=0; ii<nRows; ++ii) {
                tab_printf("Row %i:", ii+1);
                for (jj=0; jj<nCols; ++jj) {
                    printf(" %i", ivalues[ii*nCols+jj]);
                }
                printf("\n");
            }
            tab_decrement();
            break;
        default:
            printf("UNEXPECTED Parameter BaseType: %s 0x%x (%d)\n",
                      cgGetTypeString(basetype), basetype, basetype);
            break;
        }
        break;
    case CG_PARAMETERCLASS_OBJECT:
        switch (basetype) {
        case CG_STRING:
            tab_printf("Value: %s\n", cgGetStringParameterValue(parameter));
            break;
        case CG_TEXTURE:
            //<> what to do ?
            break;
        case CG_PROGRAM_TYPE:
        case CG_VERTEXSHADER_TYPE:
        case CG_PIXELSHADER_TYPE:
            // should never happen...
        default:
            printf("UNEXPECTED Parameter BaseType: %s 0x%x (%d)\n",
                      cgGetTypeString(basetype), basetype, basetype);
            break;
        }
        break;
    case CG_PARAMETERCLASS_SAMPLER:
        //<> what to do ?
        break;
    case CG_PARAMETERCLASS_UNKNOWN:
        // shouldn't happen...
    default:
        printf("UNEXPECTED Parameter Class: %s 0x%x (%d)\n",
           cgGetParameterClassString(parameterclass), parameterclass, parameterclass);
        break;
    }
}
