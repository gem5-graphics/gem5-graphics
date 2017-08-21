#include <set>
#include <string>
#include <fstream> 

#include "gpu/gpgpu-sim/cuda_gpu.hh"
#include "base/misc.hh"
#include "base/debug.hh"
#include "debug/GraphicsCalls.hh"
#include "graphics/serialize_graphics.hh"
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

checkpointGraphics checkpointGraphics::SerializeObject;

void checkpointGraphics::serializeGraphicsCommand(int pid, int tid,
        uint64_t commandCode, uint8_t* buffer, uint32_t bufLen){
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
    serializeToTmpFile(&command);
    if(isWriteCommand(commandCode)){
      delete [] nBuffer;
    }
}

void checkpointGraphics::serializeToTmpFile(GraphicsCommand_t* cmd){
  if(tmpGem5PipeOutput == NULL){
    simout.remove(tmpGem5PipeFileName);
    tmpGem5PipeOutput = simout.create(tmpGem5PipeFileName);
  }
  serializeCommand(getCmdName(cmdCount++), cmd, *tmpGem5PipeOutput->stream());
}

//used to prevent simulation from executing as a result of invoking
//graphics calls while unserialization is underway
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
    serializeAll(os);
}

void checkpointGraphics::serializeAll(std::ostream &os){

    std::string name = "fbWidth";
    int fbWidth = gem5GraphicsCalls_t::getFrameBufferWidth();
    SERIALIZE_SCALAR(fbWidth);

    name = "fbHeight";
    int fbHeight = gem5GraphicsCalls_t::getFrameBufferHeight();
    SERIALIZE_SCALAR(fbHeight);

    name = "cmdCount";
    SERIALIZE_SCALAR(cmdCount);


    if(cmdCount == 0) return; //no commands to serialize

    //add graphics commands
    if(!tmpGem5PipeOutput){
      fatal("Temporary gem5Pipe output file is not defined!\n");
    }

    tmpGem5PipeOutput->stream()->flush();

    std::ifstream cmdFile(simout.resolve(tmpGem5PipeFileName).c_str());
    if(cmdFile){
      os << cmdFile.rdbuf();
      cmdFile.close();
    } else {
      fatal("cannot find gem5pipe tmp output file!\n");
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
  if(tmpGem5PipeOutput){
    simout.close(tmpGem5PipeOutput);
  }

  if(simout.isFile(tmpGem5PipeFileName)){
    simout.remove(tmpGem5PipeFileName);
  }
}

void checkpointGraphics::unserializeGraphicsState(CheckpointIn& cp){
    //clear the frame dumping folder
    gem5GraphicsCalls_t::RemoveFrameDir();
    unserializeAll(cp);
}

void checkpointGraphics::unserializeAll(CheckpointIn& cp){
    int fbWidth, fbHeight;
    std::string name = "fbWidth";
    UNSERIALIZE_SCALAR(fbWidth);
    name = "fbHeight";
    UNSERIALIZE_SCALAR(fbHeight);
    gem5GraphicsCalls_t::setFrameBufferSize(fbWidth, fbHeight);
    gem5GraphicsCalls_t::gem5GraphicsCalls.init_gem5_graphics();

    name = "cmdCount";
    int cmdCount;
    UNSERIALIZE_SCALAR(cmdCount);

    isUnserializing = true;
    for(int i=0; i<cmdCount; i++){
        unserializeCommand(getCmdName(i), cp);
    }
    isUnserializing = false;
    //invokeAll();
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
    //mCommands.push_back(newCmd);
    invokeCommand(&newCmd);
    serializeToTmpFile(&newCmd);
    if(isWriteCommand(newCmd.commandCode)){
      delete [] newCmd.buffer;
    }
}

void checkpointGraphics::invokeCommand(GraphicsCommand_t * cmd){
    //if control command nothing to be done
    if(isControlCommand(cmd->commandCode) and !isMemCommand(cmd->commandCode))
        return;

    graphicsStream * stream = graphicsStream::get(cmd->tid, cmd->pid);

    if (isWriteCommand(cmd->commandCode)) {
       stream->write(cmd->buffer, cmd->bufferLen);
    } else if(isMemCommand(cmd->commandCode)){
        CudaGPU *cudaGPU = CudaGPU::getCudaGPU(g_active_device);
        if(cmd->buffer != NULL){
            cudaGPU->setGraphicsMem(cmd->pid, (Addr)cmd->buffer, cmd->bufferLen);
        }
    } else {
        //read command
        uint8_t *temp = new uint8_t[cmd->bufferLen];
        switch (cmd->commandCode) {
            case gem5_read:
               {
                  if(cmd->bufferLen > 0){
                     stream->read(temp, cmd->bufferLen);
                  }
               }
                break;
            default:
                //should be one of the above
                panic("Unexpected command %d\n", cmd->commandCode);
                assert(0);
        }
        delete [] temp;
    }
}

bool checkpointGraphics::isWriteCommand(uint64_t commandCode){
    return (commandCode==gem5_write);
}

bool checkpointGraphics::isMemCommand(uint64_t commandCode){
   return (commandCode==gem5_graphics_mem);
}

bool checkpointGraphics::isControlCommand(uint64_t commandCode){
    return ((commandCode==gem5_graphics_mem)
            or (commandCode==gem5_block)
            or (commandCode==gem5_debug)
            or (commandCode==gem5_call_buffer_fail)
            or (commandCode==gem5_sim_active)
            or (commandCode==gem5_get_procId));
}
