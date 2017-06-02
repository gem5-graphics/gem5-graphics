#ifndef __SERIALIZE_GRAPHICS_HH__
#define __SERIALIZE_GRAPHICS_HH__

#include <vector>
#include "graphics/graphic_calls.hh"
#include "sim/serialize.hh"



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

class checkpointGraphics{
public:
    static void serializeGraphicsState (const char* graphicsFile);
    static void unserializeGraphicsState(CheckpointIn& cp);
    static void serializeGraphicsCommand(int pid, int tid,
        uint64_t commandCode, uint8_t* buffer, uint32_t buffLen);
    static bool isUnserializingCp();
    ~checkpointGraphics();
private:
    //members
    static checkpointGraphics mSerializeObject;
    const static std::string section;
    std::vector<GraphicsCommand_t> mCommands;
    
    //methods
    void serializeCommand (std::string cmdName, GraphicsCommand_t* command, std::ostream &os);
    void unserializeCommand(std::string cmdName, CheckpointIn& cp);
    inline std::string getCmdName(int id);
    void serializeAll(std::ostream &os);
    void unserializeAll(CheckpointIn& cp);
    void invokeAll();
    void invokeCommand(GraphicsCommand_t* cmd);
    static bool isWriteCommand(uint64_t commandCode);
    static bool isControlCommand(uint64_t commandCode);
    static bool isMemCommand(uint64_t commandCode);
    static bool isUnserializing;
};

void serializeGraphicsState (const char* graphicsFile);
void unserializeGraphicsState(CheckpointIn * cp);

#endif
