#include <set>
#include <string>
#include <fstream> 

#include "gpu/gpgpu-sim/cuda_gpu.hh"
#include "base/misc.hh"
#include "base/debug.hh"
#include "debug/GraphicsCalls.hh"
#include "graphics/serialize_graphics.hh"
#include "graphics/libOpenglRender/render_api.h"
#include "graphics/graphicsStream.hh"
#include "graphics/gem5_graphics_calls.h"

extern unsigned g_active_device;

#undef SERIALIZE_SCALAR
#define SERIALIZE_SCALAR(scalar)     callSerializeScalar(os, name, #scalar, scalar)

#undef SERIALIZE_ENUM
#define SERIALIZE_ENUM(scalar)          callSerializeScalar(os, name, #scalar, (int)scalar)

#undef SERIALIZE_ARRAY
#define SERIALIZE_ARRAY(member, size)           \
        callSerializeArray(os, name,  #member, member, size)

#undef UNSERIALIZE_SCALAR
#define UNSERIALIZE_SCALAR(scalar)      callUnserializeScalar(cp, section, name, #scalar, scalar)

#undef UNSERIALIZE_ENUM
#define UNSERIALIZE_ENUM(scalar)                \
 do {                                           \
    int tmp;                                    \
    callUnserializeScalar(cp, section, name, #scalar, tmp); \
    fromInt(scalar, tmp);                    \
  } while (0) 


#undef UNSERIALIZE_ARRAY
#define UNSERIALIZE_ARRAY(member, size) callUnserializeArray(cp, section, name, #member, member, size)

#define CHECK_STRUCT(s_pointer) \
{ \
    if(s_pointer){ \
        paramOut(os, name, true); \
    } else { \
        paramOut(os, name, false); \
        return; \
    } \
} 


#define READ_STRUCT(ptr, type) \
{ \
    bool not_null; \
    paramIn(cp, name, not_null); \
    if(!not_null) \
        return; \
    if(ptr==NULL){ \
        ptr = new type; \
    } \
} 

std::string getVarFullName(const std::string &baseName, const std::string &name){
    std::size_t pos1 = name.find(">");
    std::size_t pos2 = name.find(".");
    std::size_t cutPos = (pos1>pos2? pos2:pos1)+1;
    std::string fullName = baseName + "." + name.substr(cutPos);
    return fullName;
}

template <typename T>
void
callSerializeScalar(std::ostream &os, const std::string &baseName, const std::string &name, const T &param){
    std::string fullName = getVarFullName(baseName, name);
    paramOut(os, fullName, param);
}

template <class T>
void
callSerializeArray(std::ostream &os, std::string &baseName, const std::string &name, const T *param, unsigned size)
{
    std::string fullName = getVarFullName(baseName, name);
    std::string status = fullName +".status";
    if(param!= NULL){
        paramOut(os, status, true);
        arrayParamOut(os, fullName, param, size);
    } else {
        paramOut(os, status, false);
    }
}


template <class T>
void
callUnserializeScalar(CheckpointIn& cp, const std::string &section, const std::string &baseName, 
        const std::string &name, T &param)
{
    std::string fullName = getVarFullName(baseName, name);
    paramIn(cp, fullName, param);
}

template <class T>
void
callUnserializeArray(CheckpointIn& cp, const std::string &section, const std::string &baseName, const std::string &name,
             T *param, unsigned size)
{
    std::string fullName = getVarFullName(baseName, name);
    std::string status = fullName +".status";
    bool isSerialized;
    paramIn(cp, status, isSerialized);
    if(isSerialized){
        if(param==NULL){
            param = new T[size];
        }
        arrayParamIn(cp, fullName, param, size);
    }
}

const std::string checkpointGraphics::section = "Graphics";
checkpointGraphics checkpointGraphics::mSerializeObject;

void checkpointGraphics::serializeGraphicsCommand(int pid, int tid,
        uint64_t commandCode, uint8_t* buffer, uint32_t bufLen){
    //giving the command a unique name
   // std::string name = "cmd" + std::to_string(mCommandsCount);
    
    uint8_t* nBuffer = NULL;
    if(isMemCommand(commandCode)){
       nBuffer = buffer;
    } else if(isWriteCommand(commandCode)){
       nBuffer = new uint8_t[bufLen];
       memcpy(nBuffer, buffer, bufLen);
    }
   
    DPRINTF(GraphicsCalls, "Serialization: adding command %d: pid=%d, tid=%d, buffer=%08llX, bufLen=%d\n",\
          commandCode, pid, tid, (uint64_t)nBuffer, bufLen);
    GraphicsCommand_t command(pid, tid, commandCode, bufLen, nBuffer);
    mSerializeObject.mCommands.push_back(command);
}

//used to prevent simulation from executing as a result of invoking
//graphics calls while unserialization is underway
bool checkpointGraphics::isUnserializing = false;
bool checkpointGraphics::isUnserializingCp(){
    return isUnserializing;
}

std::string checkpointGraphics::getCmdName(int cmdId){
    return "cmd"+std::to_string(cmdId);
}

void checkpointGraphics::serializeGraphicsState (const char* graphicsFile){
    std::ofstream os(graphicsFile);
    if (!os.is_open())
        fatal("Unable to open file %s for writing\n", graphicsFile);
    
    os << "\n[" << section << "]\n"; //section name
    
    mSerializeObject.serializeAll(os);
}

void checkpointGraphics::serializeAll(std::ostream &os){
    std::string name = "cmdCount";
    int cmdCount = mCommands.size();
    SERIALIZE_SCALAR(cmdCount);
    //serialize graphics commands 
    for(int i=0; i<cmdCount; i++){
        serializeCommand(getCmdName(i), &mCommands[i], os);
    }
}

void checkpointGraphics::serializeCommand(std::string name, GraphicsCommand_t * cmd, std::ostream &os)
{    
    SERIALIZE_SCALAR(cmd->pid);
    SERIALIZE_SCALAR(cmd->tid);
    SERIALIZE_SCALAR(cmd->commandCode);
    SERIALIZE_SCALAR(cmd->bufferLen);
    if(isWriteCommand(cmd->commandCode)){
        SERIALIZE_ARRAY(cmd->buffer, cmd->bufferLen);
    } else if(isMemCommand(cmd->commandCode)){
       uint64_t buffer = (uint64_t) cmd->buffer;
       SERIALIZE_SCALAR(buffer);
    }
}

checkpointGraphics::~checkpointGraphics(){
    for(int i=0; i<mCommands.size(); i++){
        if(mCommands[i].buffer and !isMemCommand(mCommands[i].commandCode))
            delete [] mCommands[i].buffer;
    }
}

void checkpointGraphics::unserializeGraphicsState(CheckpointIn& cp){
    //initialize translation library and screen 
    init_gem5_graphics(); 
    mSerializeObject.unserializeAll(cp);
}

void checkpointGraphics::unserializeAll(CheckpointIn& cp){
    //todo: map the created contexts to the older ones?
    std::string name = "cmdCount";
    int cmdCount;
    UNSERIALIZE_SCALAR(cmdCount);
  
    for(int i=0; i<cmdCount; i++){
        unserializeCommand(getCmdName(i), cp);
    }
    
    invokeAll();
}

void checkpointGraphics::unserializeCommand(std::string name, CheckpointIn& cp){
    GraphicsCommand_t newCmd;
    UNSERIALIZE_SCALAR(newCmd.pid);
    UNSERIALIZE_SCALAR(newCmd.tid);
    UNSERIALIZE_SCALAR(newCmd.commandCode);
    UNSERIALIZE_SCALAR(newCmd.bufferLen);
    if(isWriteCommand(newCmd.commandCode)){
        newCmd.buffer = new uint8_t[newCmd.bufferLen];
        UNSERIALIZE_ARRAY(newCmd.buffer, newCmd.bufferLen);
    } else if(isMemCommand(newCmd.commandCode)){ 
       uint64_t buffer;
       UNSERIALIZE_SCALAR(buffer);
       newCmd.buffer = (uint8_t*) buffer;
    } else {
        newCmd.buffer = NULL;
    }
    
    mCommands.push_back(newCmd);
}

void checkpointGraphics::invokeAll(){
    isUnserializing = true;
    for(int i=0; i<mCommands.size(); i++){
       invokeCommand(&mCommands[i]);
    }
    isUnserializing = false;
}

void checkpointGraphics::invokeCommand(GraphicsCommand_t * cmd){
    //if control command nothing to be done
    if(isControlCommand(cmd->commandCode) and !isMemCommand(cmd->commandCode))
        return;

    graphicsStream * stream = graphicsStream::get(cmd->tid, cmd->pid);

    if (isWriteCommand(cmd->commandCode)) {
       assert(SocketStream::bytesSentFromMain == 0);
       assert(SocketStream::currentMainWriteSocket == -1);
       SocketStream::bytesSentFromMain = cmd->bufferLen;
       SocketStream::currentMainWriteSocket = stream->getSocketNum();
       stream->writeFully(cmd->buffer, cmd->bufferLen);
       SocketStream::lockMainThread();
       while(!SocketStream::allRenderSocketsReady()); //wait till all other threads are waiting
    } else if(isMemCommand(cmd->commandCode)){        
        CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);
        if(cmd->buffer != NULL){
            cudaGPU->setGraphicsMem(cmd->pid, (Addr)cmd->buffer, cmd->bufferLen);
        }
    } else {
        //read command
        uint8_t *temp = new uint8_t[cmd->bufferLen];
        switch (cmd->commandCode) {
            case gem5_readFully:
               {
                  if(cmd->bufferLen > 0){
                     SocketStream::currentMainReadSocket = stream->getSocketNum();
                     stream->readFully(temp, cmd->bufferLen);
                     SocketStream::currentMainReadSocket = -1;

                     int newByteCount = SocketStream::bytesSentToMain - cmd->bufferLen;
                     assert(newByteCount >= 0);
                     bool cond = (SocketStream::bytesSentToMain == SocketStream::totalBytesSentToMain) and (cmd->bufferLen > 0);
                     if(cond){
                        SocketStream::readUnlock();
                        SocketStream::lockMainThread();
                     }
                     while(!SocketStream::allRenderSocketsReady());
                     SocketStream::bytesSentToMain = newByteCount;
                  }
               }
                break;
            default:
                //should be one of the above 
                panic("Unexpected read command");
                assert(0); 
        }
        delete [] temp;
    }
}

bool checkpointGraphics::isWriteCommand(uint64_t commandCode){
    return (commandCode==gem5_writeFully);
}

bool checkpointGraphics::isMemCommand(uint64_t commandCode){
   return (commandCode==gem5_graphics_mem);
}

bool checkpointGraphics::isControlCommand(uint64_t commandCode){
    return ((commandCode==gem5_graphics_mem)
            or (commandCode==gem5_block)
            or (commandCode==gem5_debug)
            or (commandCode==gem5_call_buffer_fail)
            or (commandCode==gem5_sim_active));
}
