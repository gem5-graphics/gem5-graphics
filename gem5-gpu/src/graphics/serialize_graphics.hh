#ifndef __SERIALIZE_GRAPHICS_HH__
#define __SERIALIZE_GRAPHICS_HH__

#include <vector>
#include "sim/serialize.hh"
#include "base/output.hh"

struct GraphicsCommand_t{
    int pid;
    int tid;
    uint64_t commandCode;
    uint32_t bufferLen;
    uint8_t* buffer;
    GraphicsCommand_t(){}
    GraphicsCommand_t(int ppid, int ptid, 
    uint64_t pcommandCode, uint32_t pbufferLen, uint8_t* pbuffer){
        pid = ppid;
        tid = ptid;
        commandCode = pcommandCode;
        bufferLen = pbufferLen;
        buffer = pbuffer;
    }
};

class checkpointGraphics {
 public:
    static checkpointGraphics SerializeObject;

    checkpointGraphics():
      section("Graphics"), tmpGem5PipeOutput(NULL), cmdCount(0), isUnserializing(false){
    }
    void serializeGraphicsState (const char* graphicsFile);
    void unserializeGraphicsState(CheckpointIn& cp);
    void serializeGraphicsCommand(int pid, int tid,
        uint64_t commandCode, uint8_t* buffer, uint32_t buffLen);
    bool isUnserializingCp();
    void serializeToTmpFile(GraphicsCommand_t*);
    ~checkpointGraphics();
private:
    //members
    const std::string section;
    const char* tmpGem5PipeFileName = "_gem5pipe.tmpOut";
    OutputStream* tmpGem5PipeOutput;
    int cmdCount;
    bool isUnserializing;

    //methods
    void serializeCommand (std::string cmdName, GraphicsCommand_t* command, std::ostream &os);
    void unserializeCommand(std::string cmdName, CheckpointIn& cp);
    void serializeAll(std::ostream &os);
    void unserializeAll(CheckpointIn& cp);
    void invokeCommand(GraphicsCommand_t* cmd);
    bool isWriteCommand(uint64_t commandCode);
    bool isControlCommand(uint64_t commandCode);
    bool isMemCommand(uint64_t commandCode);
    inline std::string getCmdName(int id);
};

void serializeGraphicsState (const char* graphicsFile);
void unserializeGraphicsState(CheckpointIn * cp);

#endif
